/*#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/outpost.hpp"
#include "autoaim_interfaces/msg/outposts.hpp"

#include "detector_dl/Armor.hpp"
#include "detector_dl/Outpost.hpp"
#include "detector_dl/Detector.hpp"
#include "detector_dl/Monocular.hpp"
#include "detector_dl/Monocular_Dart.hpp"
#include "std_msgs/msg/u_int8.hpp"

class DetectorDlNode : public rclcpp::Node {
private:
    // 图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    // 识别器
    std::unique_ptr<Detector> detector_without_dart;
    std::unique_ptr<Detector> detector_dart;

    // 单目解算（PNP）
    std::unique_ptr<Monocular> monocular_;
    // 前哨站解算 （PNP）
    std::unique_ptr<Monocular_Dart> monocular_dart;

    // 相机信息订阅
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

    // 源图像订阅
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    // Detected armor publisher
    rclcpp::Publisher<autoaim_interfaces::msg::Armors>::SharedPtr armors_pub_;

    rclcpp::Publisher<autoaim_interfaces::msg::Outposts>::SharedPtr outposts_pub_;

    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr points_num_sub_;

    int points_num = 4;
    cv::Mat pnp_image;

    // 存储参数值的成员变量
    int num_classes_;
    std::string target_colour_;
    float nms_thresh_;
    float bbox_conf_thresh_;
    int input_w_;
    int input_h_;
    int ckpt_num_;
    std::string engine_file_path_;

public:
    explicit DetectorDlNode(const rclcpp::NodeOptions &options)
        : Node("detector_dl_node", options) {
        RCLCPP_INFO(this->get_logger(), "Starting DetectorDlNode!");

        // 声明参数并存储值
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        num_classes_ = declare_parameter("4P_NUM_CLASSES", 36, param_desc);
        target_colour_ = declare_parameter("4P_TARGET_COLOUR", "RED", param_desc);
        nms_thresh_ = declare_parameter("4P_NMS_THRESH", 0.45, param_desc);
        bbox_conf_thresh_ = declare_parameter("4P_BBOX_CONF_THRESH", 0.5, param_desc);
        input_w_ = declare_parameter("4P_INPUT_W", 640, param_desc);
        input_h_ = declare_parameter("4P_INPUT_H", 640, param_desc);
        ckpt_num_ = declare_parameter("4P_CKPT_NUM", 4, param_desc);
        engine_file_path_ = declare_parameter("4P_engine_file_path",
                                              "/home/nvidia/mosas_autoaim_dl_dart/src/detector_dl/model/"
                                              "GreenDot.trt",
                                              param_desc);

        // 初始化Detector
        this->detector_without_dart = initDetector_without_dart();
        this->detector_dart = initDetector_dart();

        // Armor发布者
        this->armors_pub_ = this->create_publisher<autoaim_interfaces::msg::Armors>(
            "/detector/armors", rclcpp::SensorDataQoS());
        // Outpost发布者
        this->outposts_pub_ = this->create_publisher<autoaim_interfaces::msg::Outposts>(
            "/detector/outposts", rclcpp::SensorDataQoS());

        // 相机信息订阅
        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
                cam_info_ =
                    std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
                monocular_ =
                    std::make_unique<Monocular>(camera_info->k, camera_info->d);
                cam_info_sub_.reset(); // 停止接收
                monocular_dart =
                    std::make_unique<Monocular_Dart>(camera_info->k, camera_info->d);
                cam_info_sub_.reset(); // 停止接收
            });

        // data subscribe
        points_num_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
            "/points_num", rclcpp::SensorDataQoS(),
            std::bind(&DetectorDlNode::pointsNumCallback, this, std::placeholders::_1));

        // 相机图片订阅
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&DetectorDlNode::imageCallback, this, std::placeholders::_1));
    }
    ~DetectorDlNode() = default;

    void pnp_test(cv::Mat rvec, cv::Mat tvec) {
        // 3D 坐标点
        std::vector<cv::Point3f> axisPoints;
        axisPoints.push_back(cv::Point3f(0, 0, 0));  // 原点
        axisPoints.push_back(cv::Point3f(1, 0, 0));  // X 轴
        axisPoints.push_back(cv::Point3f(0, 1, 0));  // Y 轴
        axisPoints.push_back(cv::Point3f(0, 0, 1));  // Z 轴

        // 2D 图像点
        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(axisPoints, rvec, tvec, monocular_dart->CAMERA_MATRIX, monocular_dart->DISTORTION_COEFF, imagePoints);

        // 绘制坐标轴
        cv::line(this->pnp_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);  // X 轴为红色
        cv::line(this->pnp_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2);  // Y 轴为绿色
        cv::line(this->pnp_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2);  // Z 轴为蓝色
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
        std::vector<Outpost> outposts;
        auto target = detectOutposts(img_msg);
        if (points_num == 4) {
            outposts = target;

            // 当无前哨站，则不进行下一步
            if (outposts.empty()) {
                return;
            }

            // 当单目PnP生成成功
            if (monocular_dart != nullptr) {
                autoaim_interfaces::msg::Outposts outposts_msg;
                outposts_msg.header = img_msg->header;
                outposts_msg.header.frame_id = "gimbal";
                outposts_msg.outposts.reserve(outposts.size());

                // 对所有前哨站进行pnp解算
                for (const auto& outpost : outposts) {
                    auto [rVec, tVec, distance_to_image_center] =
                        monocular_dart->PnP_solver(outpost);
                    autoaim_interfaces::msg::Outpost outpost_msg;

                    // 对获得的rvec tvec对应的坐标系做旋转变换
                    // 让x轴朝前
                    cv::Mat R_x = (cv::Mat_<double>(3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
                    cv::Mat R_temp, T_temp;
                    cv::Rodrigues(rVec, R_temp);

                    R_temp = R_x * R_temp;
                    T_temp = R_x * tVec;

                    outpost_msg.pose.position.x = T_temp.at<double>(0);
                    outpost_msg.pose.position.y = T_temp.at<double>(1);
                    outpost_msg.pose.position.z = T_temp.at<double>(2);

                    // 旋转矩阵转四元数
                    tf2::Matrix3x3 tf2_rotation_matrix(
                        R_temp.at<double>(0, 0), R_temp.at<double>(0, 1),
                        R_temp.at<double>(0, 2), R_temp.at<double>(1, 0),
                        R_temp.at<double>(1, 1), R_temp.at<double>(1, 2),
                        R_temp.at<double>(2, 0), R_temp.at<double>(2, 1),
                        R_temp.at<double>(2, 2));
                    tf2::Quaternion tf2_q;
                    tf2_rotation_matrix.getRotation(tf2_q);
                    outpost_msg.pose.orientation = tf2::toMsg(tf2_q);
                    outposts_msg.outposts.push_back(outpost_msg);
                }
                outposts_pub_->publish(outposts_msg);
            }
            else {
                RCLCPP_ERROR_STREAM(this->get_logger(), "PnP init failed!");
            }
        }
    }

    std::unique_ptr<Detector> initDetector_without_dart() {
        // 装甲板限定阈值
        Detector::ArmorParams a_params;
        a_params.min_large_center_distance =
            declare_parameter("armor.min_large_center_distance", 3.2);

        // 初始化识别器
        return std::make_unique<Detector>(num_classes_, target_colour_, nms_thresh_, bbox_conf_thresh_, input_w_, input_h_, ckpt_num_, engine_file_path_, a_params);
    }

    std::unique_ptr<Detector> initDetector_dart() {
        // 装甲板限定阈值-无用
        Detector::ArmorParams a;

        // 初始化识别器
        return std::make_unique<Detector>(num_classes_, target_colour_, nms_thresh_, bbox_conf_thresh_, input_w_, input_h_, ckpt_num_, engine_file_path_, a);
    }

    std::vector<Outpost> detectOutposts(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) {
        // 转换为Mat
        auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
        pnp_image = img;
        auto start_time = this->now();

        points_num = 4;
        if (points_num == 4) {
            detector_dart->NUM_CLASSES = num_classes_;
            detector_dart->CKPT_NUM = ckpt_num_;
            detector_dart->TARGET_COLOUR = target_colour_;
            detector_dart->NMS_THRESH = nms_thresh_;
            detector_dart->BBOX_CONF_THRESH = bbox_conf_thresh_;
            detector_dart->INPUT_W = input_w_;
            detector_dart->INPUT_H = input_h_;

            RCLCPP_INFO_STREAM(this->get_logger(), "TARGET_COLOUR: " << detector_dart->TARGET_COLOUR);
            RCLCPP_INFO_STREAM(this->get_logger(), "CKPT_NUM: " << detector_dart->CKPT_NUM);
            RCLCPP_INFO_STREAM(this->get_logger(), "NUM_CLASSES: " << detector_dart->NUM_CLASSES);
            RCLCPP_INFO_STREAM(this->get_logger(), "engine_file_path_4p: " << detector_dart->engine_file_path);
        }

        // 开始识别
        bool show_img = true;
        auto outpost = detector_dart->detect(img, show_img);
        std::vector<Outpost> outposts;
        try {
            outposts = std::get<std::vector<Outpost>>(outpost);
        }
        catch (const std::bad_variant_access&) {
            RCLCPP_ERROR(this->get_logger(), "Unexpected target type");
        }

        // 计算每张延时与FPS
        auto final_time = this->now();
        auto latency = (final_time - start_time).seconds() * 1000;
        RCLCPP_INFO_STREAM(this->get_logger(), "Latency: " << latency << "ms");
        RCLCPP_INFO_STREAM(this->get_logger(), "FPS: " << 1000 / latency);

        // 绘出图像并显示
        cv::resize(img, img, { 640, 640 });
        cv::imshow("result", img);
        cv::waitKey(1);

        return outposts;
    }

    void pointsNumCallback(const std_msgs::msg::UInt8::ConstSharedPtr points_num_msg)
    {
        RCLCPP_INFO(this->get_logger(), "[massage from dk----------->]Receive_points_num: %d  ++++++++++++++++", points_num_msg->data);

        if (points_num_msg->data == 4)
        {
            points_num = 4;
        }
        else if (points_num_msg->data == 5)
        {
            points_num = 5;
        }
    }
};
*/    
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/outpost.hpp"
#include "autoaim_interfaces/msg/outposts.hpp"

#include "detector_dl/Armor.hpp"
#include "detector_dl/Outpost.hpp"
#include "detector_dl/Detector.hpp"
#include "detector_dl/Monocular.hpp"
#include "detector_dl/Monocular_Dart.hpp"
#include "std_msgs/msg/u_int8.hpp"

class DetectorDlNode : public rclcpp::Node {
private:
    // 图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    // 识别器
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<Detector> detector_dart;

    // 单目解算（PNP）
    std::unique_ptr<Monocular> monocular_;
    // 前哨站解算 （PNP）
    std::unique_ptr<Monocular_Dart> monocular_dart;

    // 相机信息订阅
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

    // 源图像订阅
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    // Detected armor publisher
    rclcpp::Publisher<autoaim_interfaces::msg::Armors>::SharedPtr armors_pub_;

    rclcpp::Publisher<autoaim_interfaces::msg::Outposts>::SharedPtr outposts_pub_;

    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr points_num_sub_;

    int points_num = 4;
    cv::Mat pnp_image;

    // 存储参数值的成员变量
    int num_classes_;
    std::string target_colour_;
    float nms_thresh_;
    float bbox_conf_thresh_;
    int input_w_;
    int input_h_;
    int ckpt_num_;
    std::string engine_file_path_;
    float armor_min_large_center_distance_;

public:
    explicit DetectorDlNode(const rclcpp::NodeOptions &options)
        : Node("detector_dl_node", options) {
        RCLCPP_INFO(this->get_logger(), "Starting DetectorDlNode!");

        // 声明参数并存储值
        rcl_interfaces::msg::ParameterDescriptor param_desc;
        num_classes_ = declare_parameter("4P_NUM_CLASSES", 36, param_desc);
        target_colour_ = declare_parameter("4P_TARGET_COLOUR", "RED", param_desc);
        nms_thresh_ = declare_parameter("4P_NMS_THRESH", 0.45, param_desc);
        bbox_conf_thresh_ = declare_parameter("4P_BBOX_CONF_THRESH", 0.5, param_desc);
        input_w_ = declare_parameter("4P_INPUT_W", 640, param_desc);
        input_h_ = declare_parameter("4P_INPUT_H", 640, param_desc);
        ckpt_num_ = declare_parameter("4P_CKPT_NUM", 4, param_desc);
        engine_file_path_ = declare_parameter("4P_engine_file_path",
                                              "/home/nvidia/mosas_autoaim_dl_dart/src/detector_dl/model/"
                                              "GreenDot.trt",
                                              param_desc);
        armor_min_large_center_distance_ = declare_parameter("armor.min_large_center_distance", 3.2);

        // 初始化Detector
        this->detector_ = initDetector();
        this->detector_dart = initDetector_dart();

        // Armor发布者
        this->armors_pub_ = this->create_publisher<autoaim_interfaces::msg::Armors>(
            "/detector/armors", rclcpp::SensorDataQoS());
        // Outpost发布者
        this->outposts_pub_ = this->create_publisher<autoaim_interfaces::msg::Outposts>(
            "/detector/outposts", rclcpp::SensorDataQoS());

        // 相机信息订阅
        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
                cam_info_ =
                    std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
                monocular_ =
                    std::make_unique<Monocular>(camera_info->k, camera_info->d);
                cam_info_sub_.reset(); // 停止接收
                monocular_dart =
                    std::make_unique<Monocular_Dart>(camera_info->k, camera_info->d);
                cam_info_sub_.reset(); // 停止接收
            });

        // data subscribe
        points_num_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
            "/points_num", rclcpp::SensorDataQoS(),
            std::bind(&DetectorDlNode::pointsNumCallback, this, std::placeholders::_1));

        // 相机图片订阅
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&DetectorDlNode::imageCallback, this, std::placeholders::_1));
    }
    ~DetectorDlNode() = default;

    void pnp_test(cv::Mat rvec, cv::Mat tvec) {
        // 3D 坐标点
        std::vector<cv::Point3f> axisPoints;
        axisPoints.push_back(cv::Point3f(0, 0, 0));  // 原点
        axisPoints.push_back(cv::Point3f(1, 0, 0));  // X 轴
        axisPoints.push_back(cv::Point3f(0, 1, 0));  // Y 轴
        axisPoints.push_back(cv::Point3f(0, 0, 1));  // Z 轴

        // 2D 图像点
        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(axisPoints, rvec, tvec, monocular_dart->CAMERA_MATRIX, monocular_dart->DISTORTION_COEFF, imagePoints);

        // 绘制坐标轴
        cv::line(this->pnp_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);  // X 轴为红色
        cv::line(this->pnp_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2);  // Y 轴为绿色
        cv::line(this->pnp_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2);  // Z 轴为蓝色
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
        std::vector<Outpost> outposts;
        auto target = detectOutposts(img_msg);
        if (points_num == 4) {
            outposts = target;

            // 当无前哨站，则不进行下一步
            if (outposts.empty()) {
                return;
            }

            // 当单目PnP生成成功
            if (monocular_dart != nullptr) {
                autoaim_interfaces::msg::Outposts outposts_msg;
                outposts_msg.header = img_msg->header;
                outposts_msg.header.frame_id = "gimbal";
                outposts_msg.outposts.reserve(outposts.size());

                // 对所有前哨站进行pnp解算
                for (const auto& outpost : outposts) {
                    auto [rVec, tVec, distance_to_image_center] =
                        monocular_dart->PnP_solver(outpost);
                    autoaim_interfaces::msg::Outpost outpost_msg;

                    // 对获得的rvec tvec对应的坐标系做旋转变换
                    // 让x轴朝前
                    cv::Mat R_x = (cv::Mat_<double>(3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
                    cv::Mat R_temp, T_temp;
                    cv::Rodrigues(rVec, R_temp);

                    R_temp = R_x * R_temp;
                    T_temp = R_x * tVec;

                    outpost_msg.pose.position.x = T_temp.at<double>(0);
                    outpost_msg.pose.position.y = T_temp.at<double>(1);
                    outpost_msg.pose.position.z = T_temp.at<double>(2);

                    // 旋转矩阵转四元数
                    tf2::Matrix3x3 tf2_rotation_matrix(
                        R_temp.at<double>(0, 0), R_temp.at<double>(0, 1),
                        R_temp.at<double>(0, 2), R_temp.at<double>(1, 0),
                        R_temp.at<double>(1, 1), R_temp.at<double>(1, 2),
                        R_temp.at<double>(2, 0), R_temp.at<double>(2, 1),
                        R_temp.at<double>(2, 2));
                    tf2::Quaternion tf2_q;
                    tf2_rotation_matrix.getRotation(tf2_q);
                    outpost_msg.pose.orientation = tf2::toMsg(tf2_q);
                    outposts_msg.outposts.push_back(outpost_msg);
                }
                outposts_pub_->publish(outposts_msg);
            }
            else {
                RCLCPP_ERROR_STREAM(this->get_logger(), "PnP init failed!");
            }
        }
    }

    std::unique_ptr<Detector> initDetector() {
        // 装甲板限定阈值
        Detector::ArmorParams a_params;
        a_params.min_large_center_distance = armor_min_large_center_distance_;

        // 初始化识别器
        return std::make_unique<Detector>(num_classes_, target_colour_, nms_thresh_, bbox_conf_thresh_, input_w_, input_h_, ckpt_num_, engine_file_path_, a_params);
    }

    std::unique_ptr<Detector> initDetector_dart() {
        // 装甲板限定阈值-无用
        Detector::ArmorParams a;

        // 初始化识别器
        return std::make_unique<Detector>(num_classes_, target_colour_, nms_thresh_, bbox_conf_thresh_, input_w_, input_h_, ckpt_num_, engine_file_path_, a);
    }

    std::vector<Outpost> detectOutposts(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) {
        // 转换为Mat
        auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
        pnp_image = img;
        auto start_time = this->now();

        points_num = 4;
        if (points_num == 4) {
            detector_dart->NUM_CLASSES = num_classes_;
            detector_dart->CKPT_NUM = ckpt_num_;
            detector_dart->TARGET_COLOUR = target_colour_;
            detector_dart->NMS_THRESH = nms_thresh_;
            detector_dart->BBOX_CONF_THRESH = bbox_conf_thresh_;
            detector_dart->INPUT_W = input_w_;
            detector_dart->INPUT_H = input_h_;

            RCLCPP_INFO_STREAM(this->get_logger(), "TARGET_COLOUR: " << detector_dart->TARGET_COLOUR);
            RCLCPP_INFO_STREAM(this->get_logger(), "CKPT_NUM: " << detector_dart->CKPT_NUM);
            RCLCPP_INFO_STREAM(this->get_logger(), "NUM_CLASSES: " << detector_dart->NUM_CLASSES);
            RCLCPP_INFO_STREAM(this->get_logger(), "engine_file_path_4p: " << detector_dart->engine_file_path);
        }

        // 开始识别
        bool show_img = true;
        auto outpost = detector_dart->detect(img, show_img);
        std::vector<Outpost> outposts;
        try {
            outposts = std::get<std::vector<Outpost>>(outpost);
        }
        catch (const std::bad_variant_access&) {
            RCLCPP_ERROR(this->get_logger(), "Unexpected target type");
        }

        // 计算每张延时与FPS
        auto final_time = this->now();
        auto latency = (final_time - start_time).seconds() * 1000;
        RCLCPP_INFO_STREAM(this->get_logger(), "Latency: " << latency << "ms");
        RCLCPP_INFO_STREAM(this->get_logger(), "FPS: " << 1000 / latency);

        // 绘出图像并显示
        cv::resize(img, img, { 640, 640 });
        cv::imshow("result", img);
        cv::waitKey(1);

        return outposts;
    }

    void pointsNumCallback(const std_msgs::msg::UInt8::ConstSharedPtr points_num_msg)
    {
        RCLCPP_INFO(this->get_logger(), "[massage from dk----------->]Receive_points_num: %d  ++++++++++++++++", points_num_msg->data);

        if (points_num_msg->data == 4)
        {
            points_num = 4;
        }
        else if (points_num_msg->data == 5)
        {
            points_num = 5;
        }
    }
};