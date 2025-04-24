#pragma once
#include "detector_dl/Armor.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

class Monocular_Dart {
public:
    // 相机参数
    cv::Mat CAMERA_MATRIX;    // 相机内参 3*3
    cv::Mat DISTORTION_COEFF; // 畸变系数 1*5

    // 物体在三维坐标系下的坐标
    std::vector<cv::Point3f> OUTPOST_POINTS_3D;     // 前哨站
    //std::vector<cv::Point3f> BASE_POINTS_3D;        // Base

    // 前哨站绿点参数 单位:mm
    // PnP重要依据,非必要无需修改
    static constexpr float OUTPOST_POINT_WIDTH = 225;//参数设定正方形
    static constexpr float OUTPOST_POINT_HEIGHT = 55;

    static constexpr float to_angle = 180.0 / CV_PI;

public:
    Monocular_Dart(const std::array<double, 9> &camera_matrix,
                   const std::vector<double> &dist_coeffs)
            : CAMERA_MATRIX(
                  cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data()))
                      .clone()),
              DISTORTION_COEFF(
                  cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data()))
                      .clone()) {
        // 单位:米 注意注意注意是米不是毫米
        constexpr double outpost_half_y = OUTPOST_POINT_WIDTH / 2.0 / 1000.0;
        constexpr double outpost_half_z = OUTPOST_POINT_HEIGHT / 2.0 / 1000.0;

        // 从左下开始,顺时针顺序
        // 坐标系: X:前 Y:左 Z:上 右手系

        OUTPOST_POINTS_3D.emplace_back(
                cv::Point3f(0, outpost_half_y, -outpost_half_z));
        OUTPOST_POINTS_3D.emplace_back(
                cv::Point3f(0, outpost_half_y, outpost_half_z));
        OUTPOST_POINTS_3D.emplace_back(
                cv::Point3f(0, -outpost_half_y, outpost_half_z));
        OUTPOST_POINTS_3D.emplace_back(
                cv::Point3f(0, -outpost_half_y, -outpost_half_z));

    }

    /**
     * @brief 单目PnP测距
     *
     * @param armor 目标装甲板（包括前哨站和Base）
     * @return std::tuple<cv::Mat, cv::Mat, double> rVec,tVec,distance 旋转矩阵
     * 平移矩阵 距离
     */
    std::tuple<cv::Mat, cv::Mat, double> PnP_solver(const Outpost &outpost) {
        // 旋转矩阵 平移矩阵
        // s[R|t]=s'  s->world coordinate;s`->camera coordinate
        cv::Mat rVec;
        cv::Mat tVec; // 目标中心的世界坐标系坐标

        // 进行PnP解算
        std::vector<cv::Point3f> object_points = OUTPOST_POINTS_3D;

        cv::solvePnP(object_points, outpost.outpostVertices_vector, CAMERA_MATRIX,
                     DISTORTION_COEFF, rVec, tVec, false, cv::SOLVEPNP_IPPE);

        // 获取前哨站中心世界系坐标
        double x_pos = tVec.at<double>(0, 0);
        double y_pos = tVec.at<double>(1, 0);
        double z_pos = tVec.at<double>(2, 0);
        // 可以根据需求决定是否打印
        // std::cout << "x_pos: " << x_pos << " y_pos: " << y_pos << " z_pos: " << z_pos << std::endl;

        // 光心距目标中心距离
        float cx = CAMERA_MATRIX.at<double>(0, 2);
        float cy = CAMERA_MATRIX.at<double>(1, 2);
        double distance = cv::norm(outpost.center - cv::Point2f(cx, cy));

        return {rVec, tVec, distance};
    }
};