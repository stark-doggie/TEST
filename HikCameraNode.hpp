#include "MvCameraControl.h"

#include "camera_info_manager/camera_info_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <image_transport/image_transport.hpp>

class HikCameraNode : public rclcpp::Node
{
private:
  sensor_msgs::msg::Image image_msg_;            // 图像信息
  image_transport::CameraPublisher camera_pub_;  // 图像信息发布者
  sensor_msgs::msg::CameraInfo camera_info_msg_; // 相机信息
  std::string camera_name_;                      // 相机名称

  std::array<double, 9> camera_matrix_;
  std::vector<double> distortion_coeff_;
  std::string distortion_model_;

  OnSetParametersCallbackHandle::SharedPtr
      params_callback_handle_; // 参数回调函数，改变参数时自动调用
private:
  std::thread capture_thread_;             // 取流线程
  int nRet = MV_OK;                        // 相机状态标志位
  MV_IMAGE_BASIC_INFO img_info_;           // 相机图像信息
  MV_CC_PIXEL_CONVERT_PARAM ConvertParam_; // 图像转换属性
  void *camera_handle_;                    // 相机把柄

public:
  explicit HikCameraNode(const rclcpp::NodeOptions &options)
      : Node("hik_camera", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting HikCameraNode!");
    // 枚举相机并绑定
    MV_CC_DEVICE_INFO_LIST DeviceList;
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &DeviceList);
    RCLCPP_INFO(this->get_logger(), "Found camera count = %d",
                DeviceList.nDeviceNum);
    while (DeviceList.nDeviceNum == 0 && rclcpp::ok())
    {
      RCLCPP_ERROR(this->get_logger(), "No camera found!");
      RCLCPP_INFO(this->get_logger(), "Enum state: [%x]", nRet);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &DeviceList);
    }
    MV_CC_CreateHandle(&camera_handle_, DeviceList.pDeviceInfo[0]);
    MV_CC_OpenDevice(camera_handle_);
    // 获取相机信息
    MV_CC_GetImageInfo(camera_handle_, &img_info_);
    image_msg_.data.reserve(img_info_.nHeightMax * img_info_.nWidthMax *
                            3); // BGR三通道 高*宽*三通道
    // 初始化转换参数
    ConvertParam_.nWidth = img_info_.nWidthMax;
    ConvertParam_.nHeight = img_info_.nHeightMax;
    ConvertParam_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
    // 指定QoS
    bool use_sensor_data_qos =
        this->declare_parameter("use_sensor_data_qos", false);
    // 第一种实时性强，第二种稳定
    auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data
                                   : rmw_qos_profile_default;
    camera_pub_ =
        image_transport::create_camera_publisher(this, "/image_raw", qos);
    // 声明相机参数并设置
    declareParams();
    // 开始取流
    MV_CC_StartGrabbing(camera_handle_);
    // 加载相机信息
    camera_name_ = this->declare_parameter("camera_name", "hik_camera");
    auto camera_matrix_vector_ = this->declare_parameter<std::vector<double>>("camera_matrix", 
        {2358.125070, 0.000000, 743.213621, 
        0.000000, 2357.300407, 563.210311, 
        0.000000, 0.000000, 1.000000}); // ROS2参数服务器不支持std::array类型的参数，因此需要进行转换
    if (camera_matrix_vector_.size() == 9) {
      std::copy(camera_matrix_vector_.begin(), camera_matrix_vector_.end(), camera_matrix_.begin());
    } 
    else {
      RCLCPP_ERROR(this->get_logger(), "Vector size does not match array size!");
      rclcpp::shutdown();
    }
    distortion_model_ = this->declare_parameter("distortion_model", "plumb_bob");
    distortion_coeff_ = this->declare_parameter<std::vector<double>>("distortion_coeff", 
        {-0.083754, 0.222157, 0.000000, 0.000000, 0.109514});

    camera_info_msg_.k = camera_matrix_;
    camera_info_msg_.distortion_model = distortion_model_;
    camera_info_msg_.d = distortion_coeff_;
    
    // 绑定改变参数需要调用的回调函数
    params_callback_handle_ = this->add_on_set_parameters_callback(std::bind(
        &HikCameraNode::parametersCallback, this, std::placeholders::_1));
    // 创建取图线程
    capture_thread_ = std::thread{[this]() -> void
                                  {
                                    MV_FRAME_OUT OutFrame;
                                    RCLCPP_INFO(this->get_logger(), "Publishing image!");
                                    image_msg_.header.frame_id = "camera_optical_frame";
                                    image_msg_.encoding = "rgb8";
                                    int fail_count = 0;
                                    while (rclcpp::ok())
                                    {
                                      nRet = MV_CC_SetCommandValue(camera_handle_, "TriggerSoftware");
                                      if (MV_OK != nRet)
                                      {
                                        // RCLCPP_INFO(this->get_logger(),"failed in TriggerSoftware[%x]\n", nRet);
                                        continue;
                                      }
                                      this->nRet = MV_CC_GetImageBuffer(camera_handle_, &OutFrame, 5);
                                      if (nRet == MV_OK)
                                      {
                                        fail_count = 0;
                                        // RCLCPP_INFO(this->get_logger(), "GetOneFrame, nFrameNum[%d]\n", OutFrame.stFrameInfo.nFrameNum);
                                        ConvertParam_.pDstBuffer = image_msg_.data.data();
                                        ConvertParam_.nDstBufferSize = image_msg_.data.size();
                                        ConvertParam_.pSrcData = OutFrame.pBufAddr;
                                        ConvertParam_.nSrcDataLen = OutFrame.stFrameInfo.nFrameLen;
                                        ConvertParam_.enSrcPixelType = OutFrame.stFrameInfo.enPixelType;
                                        MV_CC_ConvertPixelType(camera_handle_, &ConvertParam_);
                                        camera_info_msg_.header.stamp = image_msg_.header.stamp = this->now();
                                        image_msg_.height = OutFrame.stFrameInfo.nHeight;
                                        image_msg_.width = OutFrame.stFrameInfo.nWidth;
                                        image_msg_.step = OutFrame.stFrameInfo.nWidth * 3;
                                        image_msg_.data.resize(image_msg_.width * image_msg_.height * 3);
                                        camera_pub_.publish(image_msg_, camera_info_msg_);
                                        MV_CC_FreeImageBuffer(camera_handle_, &OutFrame);
                                        continue;
                                      }
                                      else
                                      {
                                        fail_count++;
                                        if (fail_count >= 10)
                                        {
                                          RCLCPP_INFO(this->get_logger(), "Get buffer failed! nRet: [%x]", nRet);
                                          MV_CC_StopGrabbing(camera_handle_);
                                          MV_CC_StartGrabbing(camera_handle_);
                                        }
                                        else if (fail_count >= 50)
                                        {
                                          RCLCPP_FATAL(this->get_logger(), "Camera failed!");
                                          rclcpp::shutdown();
                                        }
                                      }
                                    }
                                  }};
  }
  ~HikCameraNode()
  {
    if (capture_thread_.joinable())
    {
      capture_thread_.join();
    }
    if (camera_handle_)
    {
      MV_CC_StopGrabbing(camera_handle_);
      MV_CC_CloseDevice(camera_handle_);
      MV_CC_DestroyHandle(&camera_handle_);
    }
    RCLCPP_INFO(this->get_logger(), "HikCameraNode destroyed!");
  }

  // 声明参数的各种属性
  void declareParams()
  {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    MVCC_FLOATVALUE fValue;
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;
    // 下面将对每个参数分别定义最大值，最小值，描述

    // Exposure time
    param_desc.description = "Exposure time in microseconds";
    MV_CC_GetFloatValue(camera_handle_, "ExposureTime", &fValue);
    param_desc.integer_range[0].from_value = fValue.fMin;
    param_desc.integer_range[0].to_value = fValue.fMax;
    double exposure_time =
        this->declare_parameter("exposure_time", 5000.0, param_desc);
    nRet = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
    if (MV_OK != nRet)
    {
      RCLCPP_INFO(this->get_logger(), "MV_CC_SetExposureTime fail! nRet [%x]\n", nRet);
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "Exposure time: %f", exposure_time);
    }

    // Gain
    param_desc.description = "Gain";
    MV_CC_GetFloatValue(camera_handle_, "Gain", &fValue);
    param_desc.integer_range[0].from_value = fValue.fMin;
    param_desc.integer_range[0].to_value = fValue.fMax;
    double gain = this->declare_parameter("gain", fValue.fCurValue, param_desc);
    nRet = MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
    if (MV_OK != nRet)
    {
      RCLCPP_INFO(this->get_logger(), "MV_CC_SetGain fail! nRet [%x]\n", nRet);
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "Gain: %f", gain);
    }
    // 设置触发模式为on
    // set trigger mode as on
    nRet = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 1);
    if (MV_OK != nRet)
    {
      RCLCPP_INFO(this->get_logger(), "MV_CC_SetTriggerMode fail! nRet [%x]\n", nRet);
    }
    // 设置触发源
    // set trigger source
    nRet = MV_CC_SetEnumValue(camera_handle_, "TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE);
    if (MV_OK != nRet)
    {
      RCLCPP_INFO(this->get_logger(), "MV_CC_SetTriggerSource fail! nRet [%x]\n", nRet);
    }
  }

  // 使用rqt改变参数时调用的函数
  rcl_interfaces::msg::SetParametersResult
  parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto &param : parameters)
    {
      if (param.get_name() == "exposure_time")
      {
        int status =
            MV_CC_SetFloatValue(camera_handle_, "ExposureTime", param.as_int());
        RCLCPP_INFO(this->get_logger(), "Exposure time change to: %ld",
                    param.as_int());
        if (MV_OK != status)
        {
          result.successful = false;
          result.reason =
              "Failed to set exposure time, status = " + std::to_string(status);
        }
      }
      else if (param.get_name() == "gain")
      {
        int status =
            MV_CC_SetFloatValue(camera_handle_, "Gain", param.as_double());
        RCLCPP_INFO(this->get_logger(), "Gain change to: %f",
                    param.as_double());
        if (MV_OK != status)
        {
          result.successful = false;
          result.reason =
              "Failed to set gain, status = " + std::to_string(status);
        }
      }
      else
      {
        result.successful = false;
        result.reason = "Unknown parameter: " + param.get_name();
      }
    }
    return result;
  }
};
