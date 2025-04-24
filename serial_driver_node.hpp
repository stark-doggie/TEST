#include <cmath>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <autoaim_interfaces/msg/aim_info.hpp>
#include <autoaim_interfaces/msg/target.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <serial_driver/packages.hpp>
#include <serial_driver/serial_core.hpp>
#include <serial_driver/search_serialport.hpp>

class SerialDriverNode : public rclcpp::Node {
private:
  std::unique_ptr<USBSerial> serial_core_;

  // TF broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Subscription<autoaim_interfaces::msg::AimInfo>::SharedPtr
      aim_info_sub_;

  rclcpp::Subscription<autoaim_interfaces::msg::Target>::SharedPtr
      target_sub;

  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr
      game_progress_pub_;

  autoaim_interfaces::msg::Target target;

  float RADIUS;
  std::string PORTNAME;
  std::string portname_;  // PORTNAME Parameter
  bool DEBUG;
  double timestamp_offset_;
  double cam_x, cam_z;

public:
  explicit SerialDriverNode(const rclcpp::NodeOptions &options)
      : Node("serial_driver_node", options) {
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    RCLCPP_INFO(this->get_logger(), "Starting SerialDriverNode!");

    {
    // Search for serial ports matching the pattern "/dev/ttyACM*"
    std::vector<std::string> serialPorts = findSerialPorts("ttyACM[0-9]");
    // Display the found serial ports
    if (serialPorts.empty()) {
        std::cout << "No serial ports found matching the pattern." << std::endl;
    } else {
        std::cout << "Found serial ports:" << std::endl;
        for (const auto& port : serialPorts) {
            portname_ = port;
            std::cout << port << std::endl;
        }
    }}

    this->declare_parameter("roll", 0.0);
    this->declare_parameter("pitch", 0.0);
    this->declare_parameter("yaw", 0.0);
    this->declare_parameter("timestamp_offset", 0.006);
    this->declare_parameter("cam_x", 0.1);  // 相机x轴相对于云台旋转中心的偏移量(正数)，单位m
    this->declare_parameter("cam_z", 0.05); // 相机z轴相对于云台旋转中心的偏移量(正数)，单位m
    RADIUS = this->declare_parameter("RADIUS", 0.20);
    DEBUG = this->declare_parameter("DEBUG", true);
    

    serial_core_ = std::make_unique<USBSerial>(portname_);

    this->game_progress_pub_ = this->create_publisher<std_msgs::msg::UInt8>("/game_progress", rclcpp::SensorDataQoS());

    serial_core_->start_receive_thread(std::bind(
        &SerialDriverNode::received_handler, this, std::placeholders::_1));

    target_sub = this->create_subscription<autoaim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS(),
      std::bind(&SerialDriverNode::target_callback, this,
                  std::placeholders::_1));

    // aim_info_sub_ = this->create_subscription<autoaim_interfaces::msg::AimInfo>(
    //     "/aim_info", rclcpp::SensorDataQoS(),
    //     std::bind(&SerialDriverNode::aim_info_callback, this,
    //               std::placeholders::_1));
  }
  ~SerialDriverNode() = default;

  void received_handler(const ReceivePackage &data) {

    // TEST: 发布TF
    geometry_msgs::msg::TransformStamped transform_stamped, transform_gimbal_cam;
    timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
    cam_x = this->get_parameter("cam_x").as_double();
    cam_z = this->get_parameter("cam_z").as_double();
    transform_stamped.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
    transform_stamped.header.frame_id = "odom";
    transform_stamped.child_frame_id = "gimbal";
    // ROLL PITCH YAW角遵循右手定则
    // transform_stamped.transform.translation.x =
    //     RADIUS * cos((data.pitch / 180.0 * M_PI)) *
    //     cos((data.yaw / 180.0 * M_PI));
    // transform_stamped.transform.translation.y =
    //     RADIUS * cos((data.pitch / 180.0 * M_PI)) *
    //     sin((data.yaw / 180.0 * M_PI));
    // transform_stamped.transform.translation.z =
    //     RADIUS * sin((data.pitch / 180.0 * M_PI));
    transform_stamped.transform.translation.x = 0;
    transform_stamped.transform.translation.y = 0;
    transform_stamped.transform.translation.z = 0;

    tf2::Quaternion q;
    q.setRPY(data.roll / 180.0 * M_PI, -data.pitch / 180.0 * M_PI,
             data.yaw / 180.0 * M_PI);
    transform_stamped.transform.rotation = tf2::toMsg(q);

    // 云台到相机的变换
    transform_gimbal_cam.header.stamp = transform_stamped.header.stamp;
    transform_gimbal_cam.header.frame_id = "gimbal";
    transform_gimbal_cam.child_frame_id = "camera";
    transform_gimbal_cam.transform.translation.x = cam_x;
    transform_gimbal_cam.transform.translation.z = cam_z;

    tf_broadcaster_->sendTransform(transform_stamped);
    tf_broadcaster_->sendTransform(transform_gimbal_cam);

    // 发布当前比赛阶段
    auto progress_ = std_msgs::msg::UInt8();
    progress_.data = data.Game_Stage;
    game_progress_pub_->publish(progress_);

    // TEST: 更新参数
    // this->set_parameter(rclcpp::Parameter("roll", data.roll));
    // this->set_parameter(rclcpp::Parameter("pitch", data.pitch));
    // this->set_parameter(rclcpp::Parameter("yaw", data.yaw));
    // this->set_parameter(rclcpp::Parameter("DEBUG", false));

    // // TEST: 按顺序输出获得的数据
    if (DEBUG) {
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "Received "
                             << "header: " << std::hex << (int)data.header
                             << "game_progress: " << std::hex << (int)data.Game_Stage
                             << " color: " << std::hex << (int)data.detect_color
                             << " id: " << std::hex << (int)data.target_id);
      RCLCPP_INFO(this->get_logger(), "Received roll: %f", data.roll);
      RCLCPP_INFO(this->get_logger(), "Received pitch: %f", data.pitch);
      RCLCPP_INFO(this->get_logger(), "Received yaw: %f", data.yaw);
      RCLCPP_INFO(this->get_logger(), "Received crc16: %x", data.crc16);
    }
  }

  void
  target_callback(const autoaim_interfaces::msg::Target::SharedPtr msg){
    SendPackage package;
    package.target_yaw = 0;
    package.target_pitch = 0;
    
    // TODO: 获取目标坐标
    package.target_x = msg->position.x;
    package.target_y = msg->position.y;
    package.target_z = msg->position.z;

    RCLCPP_INFO_STREAM(this->get_logger(),
                       "x: " << package.target_x << " y: " << package.target_y
                               << " z:" << package.target_z);

    serial_core_->send(package);
  }

  // void
  // aim_info_callback(const autoaim_interfaces::msg::AimInfo::SharedPtr msg) {
  //   float pitch_now, yaw_now;
  //   this->get_parameter("pitch", pitch_now);
  //   this->get_parameter("yaw", yaw_now);
  //   auto yaw = msg->yaw;
  //   auto pitch = msg->pitch;

  //   SendPackage package;
  //   package.target_yaw = 0;
  //   package.target_pitch = 0;

  //   // TODO: 获取目标坐标
  //   package.target_x = target.position.x;
  //   package.target_y = target.position.y;
  //   package.target_z = target.position.z;

  //   RCLCPP_INFO_STREAM(this->get_logger(),
  //                      "x: " << package.target_x << " y: " << package.target_y
  //                              << " z:" << package.target_z);

  //   serial_core_->send(package);
  // }
};
