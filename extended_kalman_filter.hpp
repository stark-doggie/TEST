#pragma once

#include <Eigen/Dense>
#include <functional>

class ExtendedKalmanFilter {
public:
  ExtendedKalmanFilter() = default;

  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  explicit ExtendedKalmanFilter(const VecVecFunc &f, const VecVecFunc &h,
                                const VecMatFunc &j_f, const VecMatFunc &j_h,
                                const VoidMatFunc &u_q, const VecMatFunc &u_r,
                                const Eigen::MatrixXd &P0)
      : f(f), h(h), jacobian_f(j_f), jacobian_h(j_h), update_Q(u_q),
        update_R(u_r), P_post(P0), n(P0.rows()),
        I(Eigen::MatrixXd::Identity(n, n)), x_pri(n), x_post(n) {}

  // 设置初始状态
  void setState(const Eigen::VectorXd &x0) { x_post = x0; }

  // 计算预测状态
  Eigen::MatrixXd predict() {
    F = jacobian_f(x_post), Q = update_Q();

    x_pri = f(x_post);
    P_pri = F * P_post * F.transpose() + Q;

    // 处理在下一次预测之前没有测量的情况
    x_post = x_pri;
    P_post = P_pri;

    return x_pri;
  }

  // 根据测量更新估计状态
  Eigen::MatrixXd update(const Eigen::VectorXd &z) {
    H = jacobian_h(x_pri), R = update_R(z);

    K = P_pri * H.transpose() * (H * P_pri * H.transpose() + R).inverse();
    x_post = x_pri + K * (z - h(x_pri));
    P_post = (I - K * H) * P_pri;

    return x_post;
  }

private:
  // 过程非线性向量函数
  VecVecFunc f;
  // 观测非线性向量函数
  VecVecFunc h;
  // f()的雅可比矩阵
  VecMatFunc jacobian_f;
  Eigen::MatrixXd F;
  // h()的雅可比矩阵
  VecMatFunc jacobian_h;
  Eigen::MatrixXd H;
  // 过程噪声协方差矩阵
  VoidMatFunc update_Q;
  Eigen::MatrixXd Q;
  // 测量噪声协方差矩阵
  VecMatFunc update_R;
  Eigen::MatrixXd R;

  // 先验误差估计协方差矩阵
  Eigen::MatrixXd P_pri;
  // 后验误差估计协方差矩阵
  Eigen::MatrixXd P_post;

  // 卡尔曼增益
  Eigen::MatrixXd K;

  // 系统维度
  int n;

  // N维单位矩阵
  Eigen::MatrixXd I;

  // 先验状态
  Eigen::VectorXd x_pri;
  // 后验状态
  Eigen::VectorXd x_post;
};
