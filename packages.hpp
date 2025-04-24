#pragma once
#include <cstdint>
#include <vector>

struct ReceivePackage
{
    uint8_t header = 0x5A;
    uint8_t Game_Stage;
    uint8_t points_num;
    uint8_t is_large_buff;
    uint8_t detect_color = 0; //  自家车红0蓝101（参考裁判系统），目标需要反过来
    uint8_t target_id = 0x01; // 1-n 待定
    float roll;
    float pitch;
    float yaw;
    uint16_t crc16 = 0xFFFF; // crc16校验
} __attribute__((packed));


struct SendPackage
{
    uint8_t header = 0xA5;
    float target_yaw;
    float target_pitch;
    float target_x;
    float target_y;
    float target_z;
    //float linear_x=1.0;  // 添加用于存储 cmd_vel 线速度 x 的成员
    //float angular_z=1.0; // 添加用于存储 cmd_vel 角速度 z 的成员
    uint8_t flag = 1;   //send to STM32F407
    uint16_t crc16 = 0xFFFF; // crc16校验
} __attribute__((packed));
