#include <atomic>
#include <cstdint>
#include <fcntl.h>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <string_view>
#include <termios.h>
#include <thread>
#include <unistd.h>

#include <serial_driver/crc.hpp>
#include <serial_driver/packages.hpp>

class USBSerial {
private:
  int serial_port_;
  termios tty_old;
  uint8_t buffer_[512];
  std::thread receive_thread_;
  std::mutex send_mutex_;
  std::mutex receive_mutex_;

public:
  USBSerial(const std::string &port_name) noexcept {
    serial_port_ = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_port_ < 0) {
      std::cerr << "Error: " << strerror(errno) << std::endl;
      throw std::runtime_error("Failed to open serial port");
    }
    termios tty;
    std::memset(&tty, 0, sizeof(tty));
    if (tcgetattr(serial_port_, &tty) != 0) {
      std::cerr << "Error: " << strerror(errno) << std::endl;
      throw std::runtime_error("Failed to get serial port attributes");
    }
    tty_old = tty;

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器状态，允许从串口接收数据
    tty.c_iflag &= ~(INPCK | ISTRIP); // 禁用奇偶校验，保留8位数据位
    tty.c_cflag &= ~(PARENB | PARODD);                // 禁用奇偶校验
    tty.c_cflag &= ~CSTOPB;                           // 设置为1个停止位
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);           // 禁用软件流控
    tty.c_cflag &= ~CRTSCTS;                          // 禁用硬件流控
    tty.c_oflag &= ~OPOST;                            // 禁用输出处理
    tty.c_iflag &= ~(INLCR | IGNCR | ICRNL | IGNBRK); // 禁用输入处理
    tty.c_cflag |= CS8;                               // 设置数据位为8位
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHOK | ECHONL | ISIG |
                     IEXTEN); // 设置为原始模式
    tty.c_cc[VMIN] = 0;       // 无最小字节数
    tty.c_cc[VTIME] = 0;      // 无超时

    if (tcsetattr(serial_port_, TCSANOW, &tty) != 0) {
      std::cerr << "Error: " << strerror(errno) << std::endl;
      throw std::runtime_error("Failed to set serial port attributes");
    }
  }

  ~USBSerial() {
    tcsetattr(serial_port_, TCSANOW, &tty_old);
    close(serial_port_);
  }

  void send(const SendPackage &package) {
    std::async(std::launch::async, [this, package]() {
      uint8_t packet[sizeof(SendPackage)];
      std::copy(reinterpret_cast<const uint8_t *>(&package),
                reinterpret_cast<const uint8_t *>(&package) +
                    sizeof(SendPackage),
                packet);
      Append_CRC16_Check_Sum(packet, sizeof(SendPackage));
      std::lock_guard<std::mutex> lock(send_mutex_);
      int len = write(serial_port_, &packet, sizeof(SendPackage));
      if (len < 0) {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        throw std::runtime_error("Failed to write to serial port");
      }
    });
  }

  void
  start_receive_thread(std::function<void(const ReceivePackage &)> callback) {
    receive_thread_ = std::thread([this, callback]() {
      std::atomic_int crc_check_failed = 0;
      std::atomic_int receive_check_failed = 0;
      std::atomic_int length_check_failed = 0;
      while (true) {
        std::async(std::launch::async, [this, callback, &crc_check_failed,&receive_check_failed,&length_check_failed]() {
          std::unique_lock<std::mutex> lock(receive_mutex_);
          int len = read(serial_port_, buffer_, sizeof(ReceivePackage));
          if (len < 0) {
            std::cerr << "Error: " << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to read from serial port");
            receive_check_failed++;
            if (receive_check_failed > 500) {
              std::cerr << "Error: " << strerror(errno) << std::endl;
              rclcpp::shutdown();
            }
          }
          else{
            receive_check_failed=0;
          }
          lock.unlock();
          if (len == sizeof(ReceivePackage)) {
            length_check_failed=0;
            if (Verify_CRC16_Check_Sum(buffer_, sizeof(ReceivePackage))) {
              callback(*reinterpret_cast<ReceivePackage *>(buffer_));
              crc_check_failed = 0;
            } else {
              crc_check_failed++;
              if(crc_check_failed>500){
                std::cerr << "Error: " << "CRC failed!" << std::endl;
                rclcpp::shutdown();
              }
              else if (crc_check_failed > 50) {
                std::cerr << "Error: " << strerror(errno) << std::endl;
              }
              
            }
          }
          else{
            length_check_failed++;
            if (length_check_failed > 500) {
              std::cerr << "Error: " << strerror(errno) << std::endl;
              rclcpp::shutdown();
            }
          }
        });
      }
    });

    if (receive_thread_.joinable()) {
      receive_thread_.detach();
    }
  }
};
