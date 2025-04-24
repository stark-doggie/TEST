#include "autoaim_interfaces/msg/aim_info.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include "ballistic_solution/bullet_solver.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class BallisticSolutionNode : public rclcpp::Node {
private:
  rclcpp::Subscription<autoaim_interfaces::msg::Target>::SharedPtr target_sub;

  rclcpp::Publisher<autoaim_interfaces::msg::AimInfo>::SharedPtr info_pub;

  BulletSolver bullet_solver;  // 弹道解算器

public:
  explicit BallisticSolutionNode(const rclcpp::NodeOptions &options)
      : Node("ballistic_solution_node", options) {
    RCLCPP_INFO(this->get_logger(), "Ballistic Solution Node Started");

    target_sub = this->create_subscription<autoaim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS(),
        std::bind(&BallisticSolutionNode::position_callback, this,
                  std::placeholders::_1));
    info_pub = this->create_publisher<autoaim_interfaces::msg::AimInfo>(
        "/aim_info", rclcpp::SensorDataQoS());
  }

  ~BallisticSolutionNode() = default;

  void
  position_callback(const autoaim_interfaces::msg::Target::SharedPtr target) {
    auto aiminfo = autoaim_interfaces::msg::AimInfo();
    aiminfo.yaw = atan(target->position.y / target->position.x) * 180 / M_PI;
    bullet_solver.theta_seeker(target->position.x, target->position.y);
    aiminfo.pitch = bullet_solver.theta0;
    aiminfo.distance = sqrt(target->position.x * target->position.x +
                            target->position.y * target->position.y +
                            target->position.z * target->position.z);
    
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "yaw: " << aiminfo.yaw << " pitch: " << aiminfo.pitch
                               << " distance" << aiminfo.distance);
    info_pub->publish(aiminfo);
  }
};
