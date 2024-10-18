#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
 
class CircleController : public rclcpp::Node {
public:
    CircleController() : Node("circle_controller") {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), bind(&CircleController::publishCommand, this));
    }
 
private:
    void publishCommand() {
        static float angle = 0.0;
        geometry_msgs::msg::Twist twist;
        twist.linear.x = 0.5; // 左：圆的半径1.0会碰到边界
        twist.angular.z = 1.0 / 3.0; //左： 使用圆的圆周速度公式
 
        publisher_->publish(twist);
        angle += twist.angular.z;
        if (angle >= 20 * M_PI) {   //左：修改此处可更改转动时间
            timer_->cancel();
        }
    }
 
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};
 
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CircleController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}