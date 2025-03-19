cmake_minimum_required(VERSION 3.10)
project(ROI_Expand)
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 查找 OpenCV 和 CUDA 库
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# 手动设置 TensorRT 路径
set(TENSORRT_ROOT /usr/local/TensorRT-10.3.0.26)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIBRARY_DIR ${TENSORRT_ROOT}/lib)
set(TENSORRT_LIBRARIES nvinfer nvinfer_plugin)
# 检查 TensorRT 头文件和库文件是否存在
if(NOT EXISTS ${TENSORRT_INCLUDE_DIR}/NvInfer.h)    
message(FATAL_ERROR "TensorRT headers not found in ${TENSORRT_INCLUDE_DIR}")
endif()
foreach(LIB ${TENSORRT_LIBRARIES})    
find_library(${LIB}_LIBRARY ${LIB} PATHS ${TENSORRT_LIBRARY_DIR} NO_DEFAULT_PATH)    
if(NOT ${LIB}_LIBRARY)        
message(FATAL_ERROR "TensorRT library ${LIB} not found in ${TENSORRT_LIBRARY_DIR}")    
endif()    
list(APPEND TENSORRT_LINK_LIBRARIES ${${LIB}_LIBRARY})
endforeach()
# 添加可执行文件
add_executable(roi_expand ROI_expand.cpp)
# 设置编译选项
target_compile_options(roi_expand PRIVATE -Wall -Wextra -Wpedantic)
# 链接库
target_link_libraries(roi_expand PRIVATE    
${OpenCV_LIBS}    
${TENSORRT_LINK_LIBRARIES}    
${CUDA_LIBRARIES}
)
# 包含目录
target_include_directories(roi_expand PRIVATE    
${OpenCV_INCLUDE_DIRS}    
${TENSORRT_INCLUDE_DIR}    
${CUDA_INCLUDE_DIRS}
)
