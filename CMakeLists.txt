cmake_minimum_required(VERSION 3.10)
project(ROI_Expand)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV、TensorRT 和 CUDA 库
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

# 添加可执行文件
add_executable(roi_expand ROI_expand.cpp)

# 设置编译选项
target_compile_options(roi_expand PRIVATE -Wall -Wextra -Wpedantic)

# 链接库
target_link_libraries(roi_expand PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# 包含目录
target_include_directories(roi_expand PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# 安装目标（可选）
install(TARGETS roi_expand DESTINATION bin)    
