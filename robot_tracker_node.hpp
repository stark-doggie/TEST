#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>

#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "geometry_msgs/msg/point_stamped.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <robot_tracker/extended_kalman_filter.hpp>
#include <robot_tracker/tracker.hpp>

using namespace std::chrono_literals;
class RobotTrackerNode : public rclcpp::Node
{
private:
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  message_filters::Subscriber<autoaim_interfaces::msg::Armors> armors_sub_;
  std::shared_ptr<tf2_ros::MessageFilter<autoaim_interfaces::msg::Armors>>
      tf_filter_;
  std::string target_frame_;
  rclcpp::Publisher<autoaim_interfaces::msg::Target>::SharedPtr target_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr point_pub;

  // // 重置跟踪器服务
  // rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_tracker_srv_;

  // // 跟踪器
  // std::unique_ptr<Tracker> tracker_;

  double max_armor_distance_; // 装甲板最大距离
  double max_abs_z_;          // 装甲板最大z绝对值
  rclcpp::Time last_time_;    // 上一次接受到消息的时间
  double dt_;                 // 时间间隔

  double s2qxyz_; // 位置噪声方差
  double s2qyaw_; // 偏航角噪声方差
  double s2qr_;   // 车辆半径噪声方差

  double r_xyz_factor; // （测量）位置噪声方差
  double r_yaw;        // （测量）偏航角噪声方差

  double max_match_distance_; // 最大匹配距离
  double max_match_yaw_diff_; // 最大匹配偏航角差

  double lost_time_thres_; // 丢失时间阈值
  double tracking_thres;   // 跟踪阈值

public:
  explicit RobotTrackerNode(const rclcpp::NodeOptions &options)
      : Node("robot_tracker_node", options)
  {
    last_time_ = this->now();
    target_frame_ =
        this->declare_parameter<std::string>("target_frame", "odom");
    max_armor_distance_ =
        this->declare_parameter<double>("max_armor_distance", 10.0);
    max_abs_z_ = this->declare_parameter<double>("max_z", 1.2);

    max_match_distance_ = declare_parameter("tracker.max_match_distance", 0.15);
    max_match_yaw_diff_ = declare_parameter("tracker.max_match_yaw_diff", 1.0);

    s2qxyz_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
    s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
    s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);

    r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
    r_yaw = declare_parameter("ekf.r_yaw", 0.02);

    tracking_thres = declare_parameter("tracker.tracking_thres", 5);
    lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);

    // 初始化跟踪器
    // tracker_ =
    //     std::make_unique<Tracker>(max_match_distance_, max_match_yaw_diff_);
    // tracker_->ekf_ = initEKFFunctions();
    // tracker_->tracking_thres = tracking_thres;

    // 重置跟踪器服务
    // reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
    //     "/tracker/reset",
    //     [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    //            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    //       tracker_->tracker_state = Tracker::State::LOST;
    //       response->success = true;
    //       return;
    //     });

    // 设置tf_buffer
    std::chrono::duration<int> buffer_timeout(1);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_buffer_->setCreateTimerInterface(
        std::make_shared<tf2_ros::CreateTimerROS>(
            this->get_node_base_interface(),
            this->get_node_timers_interface()));
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 订阅装甲板信息
    armors_sub_.subscribe(this, "/detector/armors",
                          rmw_qos_profile_sensor_data);
    tf_filter_ = std::make_shared<
        tf2_ros::MessageFilter<autoaim_interfaces::msg::Armors>>(
        armors_sub_, *tf_buffer_, target_frame_, 256,
        this->get_node_logging_interface(), this->get_node_clock_interface(),
        buffer_timeout);
    tf_filter_->registerCallback(&RobotTrackerNode::armorsCallback, this);

    target_pub_ = this->create_publisher<autoaim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS());

    point_pub = this->create_publisher<geometry_msgs::msg::PointStamped>(
        "/tracker/point", 10);
  }
  ~RobotTrackerNode() {}

  void armorsCallback(const autoaim_interfaces::msg::Armors::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "111111");
    for (auto &armor : msg->armors)
    {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = msg->header;
      pose.pose = armor.pose;
      try
      {
        armor.pose = tf_buffer_->transform(pose, "odom").pose;
      }
      catch (const std::exception &e)
      {
        std::cerr << e.what() << '\n';
      }
    }

    // 清除不合理装甲板
    // 基于max_armor_distance与position.z
    msg->armors.erase(
        std::remove_if(msg->armors.begin(), msg->armors.end(),
                       [this](auto &armor)
                       {
                         return armor.pose.position.z > max_abs_z_ ||
                                std::sqrt(std::pow(armor.pose.position.x, 2) +
                                          std::pow(armor.pose.position.y, 2)) >
                                    max_armor_distance_;
                       }),
        msg->armors.end());

    // 初始化target信息
    autoaim_interfaces::msg::Target target;
    rclcpp::Time received_time = msg->header.stamp;
    // dt_ = (received_time - last_time_).seconds();
    target.header.stamp = received_time;
    target.header.frame_id = "odom";

    // 更新跟踪器
    // if (tracker_->tracker_state == Tracker::State::LOST) {
    //   tracker_->init(msg);
    // } else {
    //   tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);
    //   tracker_->update(msg);

    //   // 根据跟踪器状态发布target
    //   if (tracker_->tracker_state == Tracker::DETECTING) {
    //     // 直接发送通过pnp解算的装甲板坐标
    //     target.id = msg->armors[0].number;
    //     target.position = msg->armors[0].pose.position;
    //   } else if (tracker_->tracker_state == Tracker::TRACKING ||
    //              tracker_->tracker_state == Tracker::TEMP_LOST) {
    //     // 填写target_msg
    //     const auto &state = tracker_->target_state;
    //     target.id = tracker_->tracked_id;
    //     target.position.x = state(0);
    //     target.position.y = state(2);
    //     target.position.z = state(4);

    //     auto point_stamped_ = geometry_msgs::msg::PointStamped();
    //     point_stamped_.header.stamp = received_time;
    //     point_stamped_.header.frame_id = "odom";
    //     point_stamped_.point.x = state(0);
    //     point_stamped_.point.y = state(2);
    //     point_stamped_.point.z = state(4);
    //     point_pub->publish(point_stamped_);

    // target_msg.armors_num =
    // static_cast<int>(tracker_->tracked_armors_num); target_msg.position.x
    // = state(0); target_msg.velocity.x = state(1); target_msg.position.y =
    // state(2); target_msg.velocity.y = state(3); target_msg.position.z =
    // state(4); target_msg.velocity.z = state(5); target_msg.yaw =
    // state(6); target_msg.v_yaw = state(7); target_msg.radius_1 =
    // state(8); target_msg.radius_2 = tracker_->another_r; target_msg.dz =
    // tracker_->dz;
    // }
    // }

    // last_time_ = received_time;
    // target_pub_->publish(target);

    // if (msg->armors.size() == 0) {
    //   return;
    // } else { // TEST
    // 选择最近的装甲板
    auto min_distance_armor = std::min_element(
    msg->armors.begin(), msg->armors.end(), [](auto &a, auto &b)
    {
      double a_x0y_distance = std::sqrt(std::pow(a.pose.position.x, 2) + std::pow(a.pose.position.y, 2));
      double b_x0y_distance = std::sqrt(std::pow(b.pose.position.x, 2) + std::pow(b.pose.position.y, 2));
      // auto armer_a = a;
      // auto armer_b = b;
      //RCLCPP_INFO(rclcpp::get_logger("your_node_name"), "Comparing armors: A(number='%s', x=%f, y=%f, dist=%f), B(number='%s', x=%f, y=%f, dist=%f)",
          //armer_a.number.c_str(), a.pose.position.x, a.pose.position.y, a_x0y_distance,
          //armer_b.number.c_str(), b.pose.position.x, b.pose.position.y, b_x0y_distance);
      if(a_x0y_distance + 0.2 < b_x0y_distance) 
      {
        return true;
      } 
      else if (a_x0y_distance > b_x0y_distance + 0.2)
      {
        return false;
      } 
      else 
      {
        return a.pose.position.y > b.pose.position.y;
      }
    });
    
    target.id = min_distance_armor->number;
    target.position = min_distance_armor->pose.position;
    target_pub_->publish(target);

    auto point = geometry_msgs::msg::PointStamped();
    point.header.stamp = msg->header.stamp;
    point.header.frame_id = "odom";
    point.point = min_distance_armor->pose.position;
    point_pub->publish(point);
    //   // TEST:发布装甲板中心点
    //   geometry_msgs::msg::PointStamped point;
    //   point.header.stamp = received_time;
    //   point.header.frame_id = "odom";
    //   point.point = min_distance_armor->pose.position;
    //   point_pub->publish(point);
    // }
  }

  // std::unique_ptr<ExtendedKalmanFilter> initEKFFunctions() {
  //   // 状态转移函数
  //   // 观测x,v_x,y,v_y,z,v_z,yaw,v_yaw,r （针对于车的xyz，非装甲板xyz）
  //   auto f = [this](const Eigen::VectorXd &x) -> Eigen::VectorXd {
  //     Eigen::VectorXd x_new(9);
  //     x_new(0) = x(0) + x(1) * dt_; // x = x + v_x * dt
  //     x_new(1) = x(1);              // v_x
  //     x_new(2) = x(2) + x(3) * dt_; // y = y + v_y * dt
  //     x_new(3) = x(3);              // v_y
  //     x_new(4) = x(4) + x(5) * dt_; // z = z + v_z * dt
  //     x_new(5) = x(5);              // v_z
  //     x_new(6) = x(6) + x(7) * dt_; // yaw = yaw + v_yaw * dt
  //     x_new(7) = x(7);              // v_yaw
  //     x_new(8) = x(8);              // r
  //     return x_new;
  //   };

  //   auto JacobianF = [this](const Eigen::VectorXd &) {
  //     Eigen::MatrixXd f(9, 9);
  //     // clang-format off
  //     f <<  1,   dt_, 0,   0,   0,   0,   0,   0,   0,
  //           0,   1,   0,   0,   0,   0,   0,   0,   0,
  //           0,   0,   1,   dt_, 0,   0,   0,   0,   0,
  //           0,   0,   0,   1,   0,   0,   0,   0,   0,
  //           0,   0,   0,   0,   1,   dt_, 0,   0,   0,
  //           0,   0,   0,   0,   0,   1,   0,   0,   0,
  //           0,   0,   0,   0,   0,   0,   1,   dt_, 0,
  //           0,   0,   0,   0,   0,   0,   0,   1,   0,
  //           0,   0,   0,   0,   0,   0,   0,   0,   1;
  //     // clang-format on
  //     return f;
  //   };

  //   // 观测函数
  //   // 观测*装甲板*的x,y,z,yaw
  //   auto h = [this](const Eigen::VectorXd &x) -> Eigen::VectorXd {
  //     Eigen::VectorXd z(4);
  //     double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
  //     z(0) = xc - r * cos(yaw); // xa
  //     z(1) = yc - r * sin(yaw); // ya
  //     z(2) = x(4);              // za
  //     z(3) = x(6);              // yaw
  //     return z;
  //   };

  //   // 观测函数的雅可比矩阵
  //   auto JacobianH = [](const Eigen::VectorXd &x) {
  //     Eigen::MatrixXd h(4, 9);
  //     double yaw = x(6), r = x(8);
  //     // clang-format off
  //     //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
  //     h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
  //           0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
  //           0,   0,   0,   0,   1,   0,   0,          0,   0,
  //           0,   0,   0,   0,   0,   0,   1,          0,   0;
  //     // clang-format on
  //     return h;
  //   };

  //   // 更新过程噪声协方差矩阵函数
  //   auto update_Q = [this]() -> Eigen::MatrixXd {
  //     Eigen::MatrixXd q(9, 9);
  //     double t = dt_, x = s2qxyz_, y = s2qyaw_, r = s2qr_;
  //     double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x,
  //            q_vx_vx = pow(t, 2) * x;
  //     double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x,
  //            q_vy_vy = pow(t, 2) * y;
  //     double q_r = pow(t, 4) / 4 * r;
  //     // clang-format off
  //     //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
  //     q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
  //           q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
  //           0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
  //           0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
  //           0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
  //           0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
  //           0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
  //           0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
  //           0,      0,      0,      0,      0,      0,      0,      0,      q_r;
  //     // clang-format on
  //     return q;
  //   };

  //   // 更新观测噪声协方差矩阵函数
  //   auto update_R = [this](const Eigen::VectorXd &z) {
  //     Eigen::DiagonalMatrix<double, 4> r;
  //     double x = r_xyz_factor;
  //     r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
  //     return r;
  //   };

  //   // 误差估计协方差矩阵
  //   Eigen::DiagonalMatrix<double, 9> p0;
  //   p0.setIdentity();

  //   return std::make_unique<ExtendedKalmanFilter>(f, h, JacobianF, JacobianH,
  //                                                 update_Q, update_R, p0);
  // }
};