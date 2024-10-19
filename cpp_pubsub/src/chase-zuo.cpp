#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <random>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "turtlesim/msg/pose.hpp"
#include "turtlesim/srv/spawn.hpp"
#include "turtlesim/srv/teleport_absolute.hpp"
#include "turtlesim/srv/kill.hpp"// 新增：用于删除乌龟
#include "turtlesim/srv/set_pen.hpp"//新增：运动轨迹画笔服务


using namespace std::chrono_literals;

//墙壁位置,碰到墙壁掉头.墙壁左下角（0.1,0.1），墙壁右上角（9,9）
const double WALL_R_X=9.9;//right wall
const double WALL_L_X=0.1;//left wall
const double WALL_T_Y=9.9;//top wall
const double WALL_B_Y=0.1;//bottom wall
//声明全局变量,记录目标乌龟是第几只,Turtle2是第1只，Turtle2是第2只
extern int g_turtlecount=1; 

class TurtleChaser : public rclcpp::Node
{
public:
  TurtleChaser() : Node("turtle_chaser"),
    chaser_publisher_(this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 10)),
    target_subscriber_(this->create_subscription<turtlesim::msg::Pose>(
      "turtle1/pose", 10, std::bind(&TurtleChaser::target1_callback, this, std::placeholders::_1))),
    target2_subscriber_(this->create_subscription<turtlesim::msg::Pose>(
      "turtle2/pose", 10, std::bind(&TurtleChaser::target2_callback, this, std::placeholders::_1))),
    target2_publisher_(this->create_publisher<geometry_msgs::msg::Twist>("turtle2/cmd_vel", 10)),//新增，让目标乌龟2跑起来
    target3_publisher_(this->create_publisher<geometry_msgs::msg::Twist>("turtle3/cmd_vel", 10)),//新增，让目标乌龟3跑起来
    spawn_srv_client_(this->create_client<turtlesim::srv::Spawn>("spawn")),
    kill_srv_client_(this->create_client<turtlesim::srv::Kill>("kill"))// 新增：创建删除服务客户端
  {
    // Wait for the spawn service to be available.
    while (!spawn_srv_client_->wait_for_service(1s)) {
            RCLCPP_INFO(this->get_logger(), "Waiting for spawn service...");
        }
    // Spawn two turtles
    //spawn_turtle("turtle1", 5, 5, 0); // 0 degrees
    //spawn_turtle("turtle2", 3, 3, M_PI / 2); // 90 degrees
    spawn_random_turtle("turtle2");

    /*
    // 创建客户端以调用/set_pen服务
    client_setpen_= this->create_client<turtlesim::srv::SetPen>("set_pen");
    // 等待客户端准备好
    if (!client_setpen_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_ERROR(this->get_logger(), "Service not available after waiting for 1s. Exiting.");
        //rclcpp::shutdown();
    }
    // 设置画笔参数（例如：颜色为黑色，宽度为0）
    auto request = std::make_shared<turtlesim::srv::SetPen::Request>();
    request->off = true;  // 关闭画笔
    request->r = 0;        // 红色分量
    request->g = 0;        // 绿色分量
    request->b = 0;        // 蓝色分量
    request->width = 0;    // 宽度设置为0（如果支持的话）
    // 调用服务
    auto result_future = client_setpen_->async_send_request(request);
    // 等待结果
    RCLCPP_INFO(this->get_logger(), "Waiting for set_pen service result...");
    //turtlesim::srv::SetPen::ResponsePtr response = result_future.get();
    //if (response->success) {
    //    RCLCPP_INFO(this->get_logger(), "Successfully set pen parameters.");
    //} else {
    //    RCLCPP_ERROR(this->get_logger(), "Failed to set pen parameters.");
    //}
    */

    // Set up a timer to update the turtle velocities
    timer_ = this->create_wall_timer(
        1000ms, std::bind(&TurtleChaser::timer_callback, this));

  }

private:
  void spawn_random_turtle(const std::string& name)
  {    
    //利用当前时间生成的种子，可保证每次生成的值都不一样
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(seed);
    static std::uniform_real_distribution<double> distribution(1.0, 9.0);

    double x = distribution(generator);
    double y = distribution(generator);
    double theta = distribution(generator); // 也可以使用固定角度，这里随机化

    spawn_turtle(name, x, y, theta);
  }

  void spawn_turtle(const std::string& name, double x, double y, double theta)
  {
    auto request = std::make_shared<turtlesim::srv::Spawn::Request>();
    request->x = x;
    request->y = y;
    request->theta = theta;
    request->name = name;
    auto future = spawn_srv_client_->async_send_request(request);
    RCLCPP_INFO(this->get_logger(), "Spawning turtle '%s' at (%f, %f, %f)",
        name.c_str(), x, y, theta);
    /*future.wait();
    if (future.result()) {
        RCLCPP_INFO(this->get_logger(), "Turtle '%s' spawned successfully", name.c_str());
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to spawn turtle '%s'", name.c_str());
    }*/
  }

  void kill_turtle(const std::string& name)
  {
    auto request = std::make_shared<turtlesim::srv::Kill::Request>();
    request->name = name;
    auto future = kill_srv_client_->async_send_request(request);
    RCLCPP_INFO(this->get_logger(), "Killing turtle '%s'", name.c_str());
  }

  void target1_callback(const turtlesim::msg::Pose::SharedPtr msg)
  {
    turtle1_pose_.x = msg->x;
    turtle1_pose_.y = msg->y;
    turtle1_pose_.theta = msg->theta;
    //RCLCPP_INFO(this->get_logger(), "Turtle1 position: x=%f, y=%f", msg->x, msg->y);

  }

  void target2_callback(const turtlesim::msg::Pose::SharedPtr msg)
  {
    turtle2_pose_.x = msg->x;
    turtle2_pose_.y = msg->y;
    turtle2_pose_.theta = msg->theta;
    //RCLCPP_INFO(this->get_logger(), "Turtle2: x=%f, y=%f", msg->x, msg->y);

  }

  void timer_callback()
    {
        // Get the current positions of both turtles (this part is simplified and would
        // normally involve subscribing to the turtles' pose topics and updating their
        // positions in some way)
        // For simplicity, we'll just assume they are always moving towards each other
        // with a fixed velocity and direction.
        // Set velocities for turtles to move towards each other
        geometry_msgs::msg::Twist turtle1_vel, taget_turtle_vel;
        // Let's assume turtle1 is moving towards turtle2's starting position (2, 2)
        // and turtle2 is moving towards turtle1's starting position (0, 0).
        // In a real scenario, you would calculate the vectors based on their actual positions.
        //turtle1_vel.linear.x = -0.5; // Turtle1 moves forward
        //turtle1_vel.angular.z = 0;  // No rotation
        // Publish the velocities
        //chaser_publisher_->publish(turtle1_vel);
        ////chaser_publisher_->publish(turtle2_vel);
        ////RCLCPP_INFO(this->get_logger(), "Published velocities");

        /*
        count++;
        if(count>4){
          kill_turtle("turtle3");
          spawn_random_turtle("turtle3");
          count=0;
        }
        */

        /*
        // 计算追逐偏差
        float x_err = turtle1_pose_.x - turtle2_pose_.x;
        float y_err = turtle1_pose_.y - turtle2_pose_.y;
        float theta_err = turtle1_pose_.theta - turtle2_pose_.theta;
 
        // 根据误差调整速度指令
        geometry_msgs::msg::Twist twist;
        twist.linear.x = 0.5* x_err; // 线速度控制增益
        twist.angular.z = 0.5* theta_err; // 角速度控制增益
        */

        //碰到墙壁转弯
        if(turtle2_pose_.x>WALL_R_X || turtle2_pose_.x<WALL_L_X || turtle2_pose_.y>WALL_T_Y || turtle2_pose_.y<WALL_B_Y)
        {
          //新增，让目标乌龟2跑起来.最好判断下乌龟是否存在，此处忽略
          taget_turtle_vel.linear.x = 0.3; // Turtle2 moves
          taget_turtle_vel.angular.z = 0.5;  // rotation        
          target2_publisher_->publish(taget_turtle_vel);
          //新增，让目标乌龟3跑起来.最好判断下乌龟是否存在，此处忽略
          taget_turtle_vel.linear.x = 0.3; // Turtle3 moves
          taget_turtle_vel.angular.z = 0.5;  // No rotation        
          target3_publisher_->publish(taget_turtle_vel);
        }
        else //直行
        {
          //新增，让目标乌龟2跑起来.最好判断下乌龟是否存在，此处忽略
          taget_turtle_vel.linear.x = 1; // Turtle2 moves
          taget_turtle_vel.angular.z = 0;  // go directly        
          target2_publisher_->publish(taget_turtle_vel);
          //新增，让目标乌龟3跑起来.最好判断下乌龟是否存在，此处忽略
          taget_turtle_vel.linear.x = 1; // Turtle3 moves
          taget_turtle_vel.angular.z = 0;  // go directly          
          target3_publisher_->publish(taget_turtle_vel);
        }
        

    
        geometry_msgs::msg::Twist twist;
        RCLCPP_INFO(this->get_logger(), "Turtle1: x=%f, y=%f", turtle1_pose_.x, turtle1_pose_.y);
        RCLCPP_INFO(this->get_logger(), "Turtle2: x=%f, y=%f", turtle2_pose_.x, turtle2_pose_.y);
        double distance = std::sqrt(std::pow(turtle2_pose_.x - turtle1_pose_.x, 2) + std::pow(turtle2_pose_.y - turtle1_pose_.y, 2));
        RCLCPP_INFO(this->get_logger(), "distance=%f",distance);
        if (distance > 1.0) // 追逐距离阈值
        {
          twist.linear.x = distance * 0.4;//*0.3是线速度调整系数
          //twist.angular.z = 0.2*atan2(turtle1_pose_.y-turtle2_pose_.y, turtle1_pose_.x-turtle2_pose_.x);//direction wrong at 1st step

          //计算目标角度
          double theta_target=atan2(turtle2_pose_.y - turtle1_pose_.y, turtle2_pose_.x - turtle1_pose_.x);
          //调整角速度,计算turble1当前朝向与目标turtle2朝向的夹角
          double delta_theta=theta_target - turtle1_pose_.theta;
          if(delta_theta > M_PI)
            delta_theta-= 2*M_PI;
          else if(delta_theta < -M_PI)
            delta_theta+= 2*M_PI;
          //根据夹角调整turtle1的角速度omega，使得turtle1逐渐朝向目标方向
          twist.angular.z= std::min(M_PI/4, delta_theta*0.5); //*0.3是角速度调整系数
        }
        else
        {
          // 当乌龟1接近乌龟2时，kill turtle2, spawn turtle3
          if(g_turtlecount==1)//target is turtle2
          {
            kill_turtle("turtle2");
            RCLCPP_INFO(this->get_logger(), "Turtle222: x=%f, y=%f", turtle2_pose_.x, turtle2_pose_.y);
            //Kill后要重新生成新的乌龟名字，否则pose信息订阅不到
            spawn_random_turtle("turtle3");
            target3_subscriber_=this->create_subscription<turtlesim::msg::Pose>(
              "turtle3/pose", 10, std::bind(&TurtleChaser::target2_callback, this, std::placeholders::_1));
            g_turtlecount=2; 
          } else  //target is turtle3
          { //kill turtle2后，turtle2的位置信息订阅不到；但是kill turtle3后能不断生成新的turtle3，并不断得到turtle3位置信息
            kill_turtle("turtle3");
            g_turtlecount=1; 
          }
          
          twist.linear.x = 0.0;
          twist.angular.z = 0.0;
        }      
        chaser_publisher_->publish(twist);


    }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr chaser_publisher_;
  rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr target_subscriber_;
  rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr target2_subscriber_;
  rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr target3_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr target2_publisher_;//新增，让目标乌龟跑起来
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr target3_publisher_;//新增，让目标乌龟跑起来
  rclcpp::Client<turtlesim::srv::Spawn>::SharedPtr spawn_srv_client_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Client<turtlesim::srv::Kill>::SharedPtr kill_srv_client_; // 新增：删除服务客户端
  turtlesim::msg::Pose turtle1_pose_; // 存储 turtle1 的位置
  turtlesim::msg::Pose turtle2_pose_; // 存储 turtle2 的位置
  rclcpp::Client<turtlesim::srv::SetPen>::SharedPtr client_setpen_;

  
  //std::string turtle1_name_;
  //std::string turtle2_name_;
};


int main(int argc, char **argv)
{
  //g_turtlecount=1;

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TurtleChaser>());
  rclcpp::shutdown();
  return 0;
}