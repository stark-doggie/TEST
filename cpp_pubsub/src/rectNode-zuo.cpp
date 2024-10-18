#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <chrono>
#include <thread>

class DrawSquare : public rclcpp::Node {
public:
  DrawSquare() : Node("draw_square") {
    // 创建一个发布者，发布到 /turtle1/cmd_vel 话题
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);

    // 创建一个定时器，每秒调用一次回调函数
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&DrawSquare::timer_callback, this)
    );

    // 初始化状态
    state_ = FORWARD;
  }

private:
  enum State { FORWARD, TURN };

  void timer_callback() {
    geometry_msgs::msg::Twist msg;

    switch (state_) {
      case FORWARD:
        msg.linear.x = 2.0;  // 线速度，正数前进
        msg.angular.z = 0.0;
        forward_count_++;
        if (forward_count_ == 2) {
          forward_count_ = 0;
          state_ = TURN;
        }
        break;
      case TURN:
        msg.linear.x = 0.0;
        msg.angular.z = M_PI / 8;  // 角速度，90度转弯
        turn_count_++;
        if (turn_count_ == 4) {
          turn_count_ = 0;
          state_ = FORWARD;
        }
        break;
    }

    RCLCPP_INFO(this->get_logger(), "Publishing: '%f', '%f'", msg.linear.x, msg.angular.z);
    publisher_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  State state_;
  int forward_count_ = 0;
  int turn_count_ = 0;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DrawSquare>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}