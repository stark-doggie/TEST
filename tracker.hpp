#include <Eigen/Eigen>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <angles/angles.h>

#include <cfloat>
#include <memory>
#include <string>

#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/target.hpp"
#include "robot_tracker/extended_kalman_filter.hpp"

using autoaim_interfaces::msg::Armor;
using autoaim_interfaces::msg::Armors;
using autoaim_interfaces::msg::Target;

enum class ArmorsNum { NORMAL_4 = 4, BALANCE_2 = 2, OUTPOST_3 = 3 };

class Tracker {
private:
  double max_match_distance_; // 最大匹配距离
  double max_match_yaw_diff_; // 最大匹配偏航角差

  int detect_count_; // 检测计数
  int lost_count_;   // 丢失计数

  double last_yaw_; // 上一次的偏航角

public:
  std::unique_ptr<ExtendedKalmanFilter> ekf_;
  int tracking_thres;
  int lost_thres;
  enum State {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;

  std::string tracked_id;
  Armor tracked_armor;
  ArmorsNum tracked_armors_num;

  double info_position_diff;
  double info_yaw_diff;

  Eigen::VectorXd measurement;

  Eigen::VectorXd target_state;

  // 用于存储另一组装甲板信息
  double dz, another_r;

public:
  Tracker(double max_match_distance, double max_match_yaw_diff)
      : tracker_state(LOST), tracked_id(std::string("")),
        measurement(Eigen::VectorXd::Zero(4)),
        target_state(Eigen::VectorXd::Zero(9)),
        max_match_distance_(max_match_distance),
        max_match_yaw_diff_(max_match_yaw_diff) {}

  ~Tracker() = default;

  void init(const Armors::SharedPtr &armors) {
    if (armors->armors.empty()) {
      return;
    }

    // 选择距离光心最近的装甲板
    double min_distance = DBL_MAX;
    tracked_armor = armors->armors.front();
    for (auto &armor : armors->armors) {
      if (armor.distance_to_image_center < min_distance) {
        min_distance = armor.distance_to_image_center;
        tracked_armor = armor;
      }
    }

    // 初始化EKF
    initEKF(tracked_armor);
    tracker_state = DETECTING;
    tracked_id = tracked_armor.number;

    // 更新装甲板应有数量
    updateArmorsNum(tracked_armor);
  }

  void update(const Armors::SharedPtr &armors) {
    // 卡尔曼预测
    auto ekf_prediction = ekf_->predict();
    // 匹配到邻近装甲板标志位
    bool matched = false;
    // 如果没有找到匹配的装甲板，则使用KF预测作为默认目标状态
    target_state = ekf_prediction;

    if (!armors->armors.empty()) {
      // 寻找与上一帧装甲板同一id最近的装甲板
      Armor same_id_armor;
      int same_id_armors_count = 0;
      auto predicted_position = getArmorPositionFromState(ekf_prediction);

      double min_position_diff = DBL_MAX;
      double yaw_diff = DBL_MAX;
      for (const auto &armor : armors->armors) {
        // 仅考虑同id装甲板
        if (armor.number == tracked_id) {
          same_id_armor = armor;
          same_id_armors_count++;
          // 计算当前装甲板与预测装甲板的位置差
          auto current_pose = armor.pose.position;
          Eigen::Vector3d current_position(current_pose.x, current_pose.y,
                                           current_pose.z);
          double position_diff = (current_position - predicted_position).norm();
          if (position_diff < min_position_diff) {
            // 更新最小位置差
            min_position_diff = position_diff;
            yaw_diff = fabs(orientationToYaw(armor.pose.orientation) -
                            ekf_prediction(6)); // ekf_prediction(6)为预测的yaw
            tracked_armor = armor;
          }
        }
      }

      // 存储位置差和偏航角差
      info_position_diff = min_position_diff;
      info_yaw_diff = yaw_diff;

      // 检查最近装甲板的距离和偏航角差是否在阈值范围内
      if (min_position_diff < max_match_distance_ &&
          yaw_diff < max_match_yaw_diff_) {
        // 成功匹配到邻近装甲板
        matched = true;
        auto tracked_armor_position = tracked_armor.pose.position;
        // 更新EKF
        double tracked_armor_yaw =
            orientationToYaw(tracked_armor.pose.orientation);
        measurement << tracked_armor_position.x, tracked_armor_position.y,
            tracked_armor_position.z, tracked_armor_yaw;
        ekf_->update(measurement);
      } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
        // 未找到匹配的装甲板，但是只有一个具有相同id的装甲板且偏航角发生了跳变，将此情况视为目标正在旋转且装甲板跳变
        handleArmorJump(same_id_armor);
      } else {
        // 未找到匹配的装甲板

      }

      // 防止半径发散 半径范围0.12~0.4
      if (target_state(8) < 0.12) {
        target_state(8) = 0.12;
        ekf_->setState(target_state);
      } else if (target_state(8) > 0.4) {
        target_state(8) = 0.4;
        ekf_->setState(target_state);
      }

      // 跟踪器状态机
      switch (tracker_state) {
      case DETECTING:
        if (matched) {
          detect_count_++;
          if (detect_count_ > tracking_thres) {
            detect_count_ = 0;
            tracker_state = TRACKING;
          }
        } else {
          detect_count_ = 0;
          tracker_state = LOST;
        }
        break;
      case TRACKING:
        if (!matched) {
          tracker_state = TEMP_LOST;
          lost_count_++;
        }
        break;
      case TEMP_LOST:
        if (!matched) {
          lost_count_++;
          if (lost_count_ > lost_thres) {
            tracker_state = LOST;
            lost_count_ = 0;
          }
        } else {
          tracker_state = TRACKING;
          lost_count_ = 0;
        }
        break;
      default:
        break;
      }
    }
  }

private:
  void initEKF(const Armor &armor) {
    double xa = armor.pose.position.x;
    double ya = armor.pose.position.y;
    double za = armor.pose.position.z;
    last_yaw_ = 0;
    double yaw = orientationToYaw(armor.pose.orientation);

    // 将初始位置设置为目标后方0.2米处
    target_state = Eigen::VectorXd::Zero(9);
    double r = 0.26;
    double xc = xa + r * cos(yaw);
    double yc = ya + r * sin(yaw);
    dz = 0, another_r = r;
    target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

    ekf_->setState(target_state);
  }
  void updateArmorsNum(const Armor &armor) {
    if (armor.type == 1 && // 仅此处：0:small 1:large
        (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
      tracked_armors_num = ArmorsNum::BALANCE_2;
    } else if (tracked_id == "outpost") {
      tracked_armors_num = ArmorsNum::OUTPOST_3;
    } else {
      tracked_armors_num = ArmorsNum::NORMAL_4;
    }
  }

  void handleArmorJump(const Armor &current_armor) {
    double yaw = orientationToYaw(
        current_armor.pose.orientation); // 此处的yaw是装甲板*朝向*的yaw
    target_state(6) = yaw;
    updateArmorsNum(current_armor);

    // 只有当装甲板数量为4时，才会出现装甲板跳变
    // 不同学校的车可能装甲板的位置，高低，距离长短都不同
    if (tracked_armors_num == ArmorsNum::NORMAL_4) {
      // target_state(4)为上一帧装甲板z坐标
      dz = target_state(4) - current_armor.pose.position.z;
      target_state(4) = current_armor.pose.position.z;
      // 更换为另一组的r
      std::swap(target_state(8), another_r);
    }

    // 如果位置差大于max_match_distance_，将其视为ekf发散，重置状态
    auto p = current_armor.pose.position;
    Eigen::Vector3d current_p(p.x, p.y, p.z);
    Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);
    if ((current_p - infer_p).norm() > max_match_distance_) {
      double r = target_state(8);
      target_state(0) = p.x + r * cos(yaw); // xc
      target_state(1) = 0;                  // vxc
      target_state(2) = p.y + r * sin(yaw); // yc
      target_state(3) = 0;                  // vyc
      target_state(4) = p.z;                // za
      target_state(5) = 0;                  // vza
    }
    ekf_->setState(target_state);
  }

  double orientationToYaw(const geometry_msgs::msg::Quaternion &q) {
    // 获得装甲板的yaw值
    tf2::Quaternion tf_q;
    tf2::fromMsg(q, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    // 使yaw的变化连续（从-pi~pi到-inf~inf）
    yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
    last_yaw_ = yaw;
    return yaw;
  }

  Eigen::Vector3d getArmorPositionFromState(const Eigen::VectorXd &x) {
    // 计算当前装甲板的预测位置
    double xc = x(0), yc = x(2), za = x(4);
    double yaw = x(6), r = x(8);
    double xa = xc - r * cos(yaw);
    double ya = yc - r * sin(yaw);
    return Eigen::Vector3d(xa, ya, za);
  }
};