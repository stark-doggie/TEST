#pragma once

#include "opencv2/opencv.hpp"
#include "bits/stdc++.h"


class Buff
{
public:
    cv::Point2f center;
    int type;
    std::vector<cv::Point2f> buffVertices_vector; // 装甲板的四个顶点 bl->tl->tr->br(向量形式) 左下 左上 右上 右下
    bool if_value = true;

public:
    Buff() = default;

    Buff(std::vector<cv::Point2f> points, int class_id) : Buff()
    {
        this->buffVertices_vector = points;
        if(class_id == 0)      //R
        {
            if(!calculate_R_value(this->buffVertices_vector[0], this->buffVertices_vector[1], 
                                  this->buffVertices_vector[2], this->buffVertices_vector[3],
                                  this->buffVertices_vector[4]))
            {
                if_value = false;
                return;
            }
            type = 1;
            calculate_R_Center(this->buffVertices_vector[0], this->buffVertices_vector[1], 
                               this->buffVertices_vector[2], this->buffVertices_vector[3],
                               this->buffVertices_vector[4]); 
        } 
        else if(class_id == 1) //AIM
        {
            if(!calculate_AIM_value(this->buffVertices_vector[0], this->buffVertices_vector[1], 
                                    this->buffVertices_vector[2], this->buffVertices_vector[3],
                                    this->buffVertices_vector[4]))
            {
                if_value = false;
                return;
            }
            type = 0;
            //解算AIM中心点暂不使用索引为3的关键点
            calculate_AIM_Center(this->buffVertices_vector[0], this->buffVertices_vector[1], 
                                 this->buffVertices_vector[2], this->buffVertices_vector[4]); 
        } 
      
    }

    ~Buff() = default;

    void calculate_AIM_Center(const cv::Point2f &right_top, const cv::Point2f &left_top,
                              const cv::Point2f &left_bottom,const cv::Point2f &right_bottom)
    {
        // 计算由 right_top 和 left_bottom 连成的直线的斜率
        double m1 = (left_bottom.y - right_top.y) / (left_bottom.x - right_top.x);
        // 计算由 right_top 和 left_bottom 连成的直线的截距
        double b1 = right_top.y - m1 * right_top.x;

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

    void calculate_R_Center(const cv::Point2f &bottom, const cv::Point2f &right_top, 
                            const cv::Point2f &left_top, const cv::Point2f &right_bottom,
                            const cv::Point2f &left_bottom)
    {
        this->center.x = (bottom.x + right_top.x + left_top.x + right_bottom.x + left_bottom.x) / 5;
        this->center.y = (bottom.y + right_top.y + left_top.y + right_bottom.y + left_bottom.y) / 5;
    }

    bool calculate_AIM_value(const cv::Point2f &right_top, const cv::Point2f &left_top,
                             const cv::Point2f &left_bottom, const cv::Point2f &bottom,
                             const cv::Point2f &right_bottom)
    {
        // 计算 right_top 到 right_bottom 的距离
        float right_distance = cv::norm(right_bottom - right_top);
        
        // 计算 left_top 到 left_bottom 的距离
        float left_distance = cv::norm(left_bottom - left_top);

        // 计算 left_bottom 到 bottom 的距离
        float left_bottom_distance = cv::norm(bottom - left_bottom);

        // 计算 right_bottom 到 bottom 的距离
        float right_bottom_distance = cv::norm(bottom - right_bottom);

        // 检查比率是否在指定范围内
        bool ratio_condition = (right_distance / left_distance >= 0.9 && right_distance / left_distance <= 1.1) &&
                            (left_bottom_distance / right_bottom_distance >= 0.9 && left_bottom_distance / right_bottom_distance <= 1.1);

        return ratio_condition;
    }

    bool calculate_R_value(const cv::Point2f &right_top, const cv::Point2f &left_top,
                           const cv::Point2f &left_bottom, const cv::Point2f &bottom,
                           const cv::Point2f &right_bottom)
    {
        // 计算每对相邻点之间的距离
        float distance_right_top_left_top = cv::norm(right_top - left_top);
        float distance_left_top_left_bottom = cv::norm(left_top - left_bottom);
        float distance_left_bottom_bottom = cv::norm(left_bottom - bottom);
        float distance_bottom_right_bottom = cv::norm(bottom - right_bottom);
        float distance_right_bottom_right_top = cv::norm(right_bottom - right_top);

        // 计算平均距离
        float average_distance = (distance_right_top_left_top + distance_left_top_left_bottom +
                                distance_left_bottom_bottom + distance_bottom_right_bottom +
                                distance_right_bottom_right_top) / 5;

        // 检查每对相邻距离是否在平均值的0.8到1.2之间
        bool within_range = (distance_right_top_left_top >= 0.8 * average_distance &&
                            distance_right_top_left_top <= 1.2 * average_distance &&
                            distance_left_top_left_bottom >= 0.8 * average_distance &&
                            distance_left_top_left_bottom <= 1.2 * average_distance &&
                            distance_left_bottom_bottom >= 0.8 * average_distance &&
                            distance_left_bottom_bottom <= 1.2 * average_distance &&
                            distance_bottom_right_bottom >= 0.8 * average_distance &&
                            distance_bottom_right_bottom <= 1.2 * average_distance &&
                            distance_right_bottom_right_top >= 0.8 * average_distance &&
                            distance_right_bottom_right_top <= 1.2 * average_distance);

        // 如果每对相邻距离都在平均值的0.9到1.1之间，则返回 true，否则返回 false
        return within_range;
    }
};