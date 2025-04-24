#pragma once

#include "opencv2/opencv.hpp"
#include "bits/stdc++.h"

class Outpost
{
public:
    cv::Point2f center;
    std::vector<cv::Point2f> outpostVertices_vector; // 前哨站的四个顶点
    bool if_value = true;

public:
    Outpost() = default;

    Outpost(std::vector<cv::Point2f> points) : Outpost()
    {
        this->outpostVertices_vector = points;
        if (!calculate_Outpost_value(this->outpostVertices_vector[0], this->outpostVertices_vector[1],
            this->outpostVertices_vector[2], this->outpostVertices_vector[3]))
        {
            if_value = false;
            return;
        }
        calculate_Outpost_Center(this->outpostVertices_vector[0], this->outpostVertices_vector[1],
            this->outpostVertices_vector[2], this->outpostVertices_vector[3]);
    }

    ~Outpost() = default;

    void calculate_Outpost_Center(const cv::Point2f& left_bottom, const cv::Point2f& left_top,
        const cv::Point2f& right_top, const cv::Point2f& right_bottom)
    {
        // 计算由 left_bottom 和 right_top 连成的直线的斜率
        double m1 = (right_top.y - left_bottom.y) / (right_top.x - left_bottom.x);
        // 计算由 left_bottom 和 right_top 连成的直线的截距
        double b1 = left_bottom.y - m1 * left_bottom.x;

        // 计算由 left_top 和 right_bottom 连成的直线的斜率
        double m2 = (right_bottom.y - left_top.y) / (right_bottom.x - left_top.x);
        // 计算由 left_top 和 right_bottom 连成的直线的截距
        double b2 = left_top.y - m2 * left_top.x;

        // 计算交点的 x 坐标
        double x = (b2 - b1) / (m1 - m2);
        // 使用其中一条直线的方程计算交点的 y 坐标
        double y = m1 * x + b1;

        this->center.x = x;
        this->center.y = y;
    }

    bool calculate_Outpost_value(const cv::Point2f& left_bottom, const cv::Point2f& left_top,
        const cv::Point2f& right_top, const cv::Point2f& right_bottom)
    {
        // 计算 left_bottom 到 left_top 的距离
        float left_distance = cv::norm(left_top - left_bottom);

        // 计算 left_top 到 right_top 的距离
        float top_distance = cv::norm(right_top - left_top);

        // 计算 right_top 到 right_bottom 的距离
        float right_distance = cv::norm(right_bottom - right_top);

        // 计算 right_bottom 到 left_bottom 的距离
        float bottom_distance = cv::norm(left_bottom - right_bottom);

        // 检查比率是否在指定范围内（这里简单假设为0.9到1.1之间，你可以根据实际情况调整）
        bool ratio_condition = (left_distance / top_distance >= 0.9 && left_distance / top_distance <= 1.1) &&
            (top_distance / right_distance >= 0.9 && top_distance / right_distance <= 1.1) &&
            (right_distance / bottom_distance >= 0.9 && right_distance / bottom_distance <= 1.1) &&
            (bottom_distance / left_distance >= 0.9 && bottom_distance / left_distance <= 1.1);

        return ratio_condition;
    }
};
