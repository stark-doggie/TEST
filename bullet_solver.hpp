#pragma once
#include <iostream>
#include <cmath>

class BulletSolver
{
public:
    float theta0;   // 需要求解的发射pitch角（角度制）
    float target_x, target_y;   // 目标的平面坐标 注意：水平x竖直y，与世界坐标系的xyz轴无关
    float u = 0.01;    // 空气阻力系数 f = kv^2 需调试
    const float g = 9.8;    // 重力加速度g
    const float v0 = 10;    // 弹丸初速度v0
    const float maxXError = 0.05;   // x方向的最大误差
    const float maxYError = 0.02;   // y方向的最大误差
    bool add_sub = true; // 增或减

public:
    explicit BulletSolver(){}

public:
    void theta_seeker(float targetX_, float targetY_)
    {
        // 根据轨迹参数方程找到正确的theta0
        theta0 = atanf(targetY_ / targetX_) * 180 / M_PI; // 初始化theta0（角度制）
        // theta0 = 0;

        // 二分法左右值
        float left = theta0, right = 45;
        int count_y = 0;
        
        while(!this->theta_checker((left + right) / 2, targetX_, targetY_)) // 采用二分法进行pitch的更新
        {
            if(add_sub)
            {
                left = (left + right) / 2;
                // std::cout << "低了" << std::endl;
            }
            else
            {
                right = (left + right) / 2;
                // std::cout << "高了" << std::endl;
            }
            count_y ++;
        }
            theta0 = (left + right) / 2;

        std::cout << "count_y = " << count_y << std::endl;
    }

public:
    bool theta_checker(float theta_temp, float targetX, float targetY)
    {
        // 将theta_temp转换为弧度制
        theta_temp = theta_temp * M_PI / 180;

        float right = tanf(theta_temp), left = -114.59;   // 二分法的左右值
        float x_temp, error_x, i_temp;

        // 计数器
        int count_i = 0;
        while(1)
        {
            // 计算x的定积分结果
            x_temp = this->integral(1, tanf(theta_temp), (left + right) / 2, theta_temp);
            // 误差在允许范围内时直接跳出循环
            error_x = fabs(x_temp - targetX);
            if(error_x <= maxXError)
            {
                i_temp = (left + right) / 2;
                break;
            }
            else if(x_temp < targetX)
            {
                right = (left + right) / 2;
                // std::cout << "偏小" << std::endl;
            }
            else
            {
                left = (left + right) / 2;
                // std::cout << "偏大" << std::endl;
            }
            count_i ++;
            if(count_i >= 50) // 若迭代次数过多
            {
                this->theta0 = atanf(targetY / targetX) * 180 / M_PI;   // 牺牲准度，保证程序不死
                return true;    // 返回真值以结束本轮解算
            }
        }
        // std::cout << "已算完x定积分并求出i值:" << x_temp << " " << i_temp << std::endl;

        // 以上过程得到的i值代入y的参数方程，计算误差
        float y_temp = this->integral(2, tanf(theta_temp), i_temp, theta_temp);  //计算y的定积分结果
        float error_y = fabs(y_temp - targetY);   //y的误差
        {
            if((y_temp < targetY))
            {
                this->add_sub = true;
                // std::cout << "特殊情况" << std::endl;
            }
            else
            {
                this->add_sub = false;
            }
        }

        // std::cout << "已算完y定积分，当前积分值和theta：" << y_temp << " " << theta_temp << std::endl;
        if(error_y <= maxYError)
        {
            // std::cout << "已找到最小误差" << std::endl;
            return true;
        }
        else
        {
            return false;
        }
    }

public:
    float Fx(float z, float theta_)
    {
        
        return 1 / 
            (u * 
                (log(z + sqrt(z * z + 1)) + 
                z * sqrt(z * z + 1) - 
                (log(tanf(theta_) + fabs(1 / cos(theta_))) + 
                    tanf(theta_) * fabs(1 / cos(theta_)) + 
                    g / (cos(theta_) * cos(theta_) * u * v0 * v0)
                )
                )
            );
    }

public:
    float Fy(float z, float theta_)
    {
        
        return z / 
            (u * 
                (log(z + sqrt(z * z + 1)) + 
                z * sqrt(z * z + 1) - 
                (log(tanf(theta_) + fabs(1 / cos(theta_))) + 
                    tanf(theta_) * fabs(1 / cos(theta_)) + 
                    g / (cos(theta_) * cos(theta_) * u * v0 * v0)
                )
                )
            );
    }

public:
    float integral(int flag, float bottom, float top, float theta_temp)
    {
        float temp =0;  //积分值
        float z = bottom;  // 积分变量 初始化为积分下限
        float delta = (top - bottom) / 1000; // 微元区间长度
        float s;    // 小矩形面积
        if(flag == 1)
        {
            //计算定积分结果
            for(int j = 1; j <= 1000; j++)
            {
                s = Fx((z + delta / 2), theta_temp) * delta;
                temp += s;
                z += delta;
            }
        }
        else if(flag == 2)
        {
            //计算定积分结果
            for(int j = 1; j <= 1000; j++)
            {
                s = Fy((z + delta / 2), theta_temp) * delta;
                temp += s;
                z += delta;
            }
        }
        else
        {
            std::cout<< "Flag error. " << std::endl;
            return -1;
        }
        return temp;
    }

};