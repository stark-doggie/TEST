cmake_minimum_required(VERSION 3.10)
project(ROI_Expand)

find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_executable(roi_expand ROI_expand.cpp)

target_link_libraries(ROI_Expand 
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)