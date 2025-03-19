cmake_minimum_required(VERSION 3.10)
project(ROI_Expand)

set(CMAKE CXX STANDARD 14)
set(CMAKE CXX STANDARD REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(TENSORRT_ROOT /usr/local/TensorRT-10.3.0.26)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIBRARY_DIR ${TENSORRT_ROOT}/lib)
set(TENSORRT_LIBRARIES nvinfer nvinfer_plugin)

if(NOT EXISTS ${TENSORRT_INCLUDE_DIR}NvInfer.h)
    message(FATAL_ERROR "TensorRT headers not found in ${TENSORRT_INCLUDE_DIR}")
endif()

foreach(LIB ${TENSORRT_LIBRARIES})
   find_library(${LIB}_LIBRARY ${LIB} PATHS ${TENSORRT_LIBRARY_DIR} NO_DEFAULT_PATH)
   if(NOT ${LIB}_LIBRARY)
       message(FATAL_ERROR "TensorRT library ${LIB} not found in ${TENSORRT_LIBRARY_DIR}")
   endif()
   list(APPEND_TENSORRT_LINK_LIBRARIES ${${LIB}_LIBRARY}) 
endforeach()   

add_executable(roi_expand ROI_expand.cpp)

target_compile_options(roi_expand PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(roi_expand PRIVATE 
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
)

target_include_directories(roi_expand PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA__INCLUDE_DIRS}
)

install(TARGETS roi_expand DESTINATION bin)
