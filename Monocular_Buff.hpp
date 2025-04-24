#pragma once
#include "detector_dl/Armor.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

class Monocular_Buff {
public:
    // 相机参数
    cv::Mat CAMERA_MATRIX;    // 相机内参 3*3
    cv::Mat DISTORTION_COEFF; // 畸变系数 1*5

    // 物体在三维坐标系下的坐标
    std::vector<cv::Point3f> R_BUFF_POINTS_3D; // R标
    std::vector<cv::Point3f> AIM_BUFF_POINTS_3D; // To-HIT标

    static constexpr double DegToRad(double angle) {
        return angle * M_PI / 180.0;
    }


    // 风车长度参数 单位:mm
    // PnP重要依据,非必要无需修改
    static constexpr float SMALL_AIM_WIDTH = 320;
    static constexpr float AIM_HEIGHT = 317;
    static constexpr float LARGE_AIM_WIDTH = 372;
    static constexpr float THIRD_POINT_DISTANCE_mm = 524.5;
    static constexpr float R_SIDE_LENGH = 88;  //68mm
   
    static constexpr float to_angle = 180.0 / CV_PI;
    // results
    // double x_pos, y_pos, z_pos; // 装甲板坐标
    // double distance;            // 相机距装甲板的距离

public:
Monocular_Buff(const std::array<double, 9> &camera_matrix,
            const std::vector<double> &dist_coeffs)
    : CAMERA_MATRIX(
            cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data()))
                .clone()),
        DISTORTION_COEFF(
            cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data()))
                .clone()) {
    // 单位:米 注意注意注意是米不是毫米
    constexpr double small_half_y = SMALL_AIM_WIDTH / 2.0 / 1000.0;
    constexpr double small_half_z = AIM_HEIGHT / 2.0 / 1000.0;
    constexpr double large_half_y = LARGE_AIM_WIDTH / 2.0 / 1000.0;
    constexpr double large_half_z = AIM_HEIGHT / 2.0 / 1000.0;
    constexpr double THIRD_POINT_DISTANCE_m = THIRD_POINT_DISTANCE_mm / 1000.0;

    static constexpr float R_DISTANCE = (R_SIDE_LENGH * sin(DegToRad(54)) / sin(DegToRad(72))) / 1000.0;

    constexpr double bottom_y = 0.0;
    constexpr double bottom_z = -R_DISTANCE;
    constexpr double right_top_y = -R_DISTANCE * sin(DegToRad(36));
    constexpr double right_top_z = R_DISTANCE * cos(DegToRad(36));
    constexpr double left_top_y = R_DISTANCE * sin(DegToRad(36));
    constexpr double left_top_z = R_DISTANCE * cos(DegToRad(36));
    constexpr double right_bottom_y = -R_DISTANCE * sin(DegToRad(36)) - R_SIDE_LENGH / 1000 * cos(DegToRad(72));
    constexpr double right_bottom_z = R_DISTANCE * cos(DegToRad(36)) - R_SIDE_LENGH / 1000 * cos(DegToRad(18));
    constexpr double left_bottom_y = R_DISTANCE * sin(DegToRad(36)) + R_SIDE_LENGH / 1000 * cos(DegToRad(72));
    constexpr double left_bottom_z = R_DISTANCE * cos(DegToRad(36)) - R_SIDE_LENGH / 1000 * cos(DegToRad(18));

    // 从右上开始,逆时针顺序
    // 坐标系: X:前 Y:左 Z:上 右手系
    AIM_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, -small_half_y, small_half_z));
    AIM_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, small_half_y, small_half_z));
    AIM_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, large_half_y, -small_half_z));
    AIM_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, 0, -THIRD_POINT_DISTANCE_m));
    AIM_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, -large_half_y, -small_half_z));
    
    
    R_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, right_top_y, right_top_z));
    R_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, left_top_y, left_top_z));
    R_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, left_bottom_y, left_bottom_z));
    R_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, bottom_y, bottom_z));
    R_BUFF_POINTS_3D.emplace_back(
        cv::Point3f(0, right_bottom_y, right_bottom_z));
    
}

/**
 * @brief 单目PnP测距
 *
 * @param buff 目标风车
 * @return std::tuple<cv::Mat, cv::Mat, double> rVec,tVec,distance 旋转矩阵
 * 平移矩阵 距离
 */
std::tuple<cv::Mat, cv::Mat, double> PnP_solver(const Buff &buff) {
    // 旋转矩阵 平移矩阵
    // s[R|t]=s'  s->world coordinate;s`->camera coordinate
    cv::Mat rVec;
    cv::Mat tVec; // 目标中心的世界坐标系坐标

    // 进行PnP解算
    auto object_points = buff.type == 1 ? R_BUFF_POINTS_3D
                                        : AIM_BUFF_POINTS_3D;

    
    cv::solvePnP(object_points, buff.buffVertices_vector, CAMERA_MATRIX,
                DISTORTION_COEFF, rVec, tVec, false, cv::SOLVEPNP_IPPE);

    // 获取目标中心世界系坐标
    double x_pos = tVec.at<double>(0, 0);
    double y_pos = tVec.at<double>(1, 0);
    double z_pos = tVec.at<double>(2, 0);
    std::cout << "x_pos: " << x_pos << "y_pos: " << y_pos << "z_pos: " << z_pos
            << std::endl;

    // 光心距目标中心距离
    float cx = CAMERA_MATRIX.at<double>(0, 2);
    float cy = CAMERA_MATRIX.at<double>(1, 2);
    double distance = cv::norm(buff.center - cv::Point2f(cx, cy));

    return {rVec, tVec, distance};
  }
};
