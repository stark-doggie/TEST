#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// 自定义的 Logger 类，用于处理 TensorRT 的日志信息
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

// 读取 .engine 文件
std::vector<char> read_engine_file(const std::string& engine_file_path) {
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_file_path << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    std::vector<char> engine_data(size);
    file.seekg(0, std::ios::beg);
    file.read(engine_data.data(), size);
    file.close();
    return engine_data;
}

// 自定义函数来尝试获取绑定维度，直接硬编码维度信息,此处是为了弥补TensorRT库没有的函数
nvinfer1::Dims getBindingDimensionsCustom(int bindingIndex) {
    nvinfer1::Dims dims;
    dims.nbDims = 3; // 输出是 3 维的
    if (bindingIndex == 1) {
        dims.d[0] = 1; // 第一维大小
        dims.d[1] = 640; // 第二维大小
        dims.d[2] = 640; // 第三维大小
    } else {
        std::cerr << "Unsupported binding index: " << bindingIndex << std::endl;
        dims.nbDims = 0;
    }
    return dims;
}

// 定义一个结构体来存储检测结果
struct DetectionResult {
    cv::Mat box; // 四个点的坐标
    float score; // 置信度分数
    int class_id; // 类别 ID
};

// 使用 TensorRT 模型进行检测
std::vector<DetectionResult> tensorrt_detection(const std::string& image_path, const std::string& engine_file_path) {
    Logger logger;
    // 创建 TensorRT 运行时对象
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return {};
    }

    // 读取 .engine 文件
    std::vector<char> engine_data = read_engine_file(engine_file_path);
    if (engine_data.empty()) {
        return {};
    }

    // 从 .engine 文件数据中反序列化生成 TensorRT 引擎
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        return {};
    }

    // 创建 TensorRT 上下文对象
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create TensorRT execution context." << std::endl;
        return {};
    }

    // 读取输入图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return {};
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // 准备输入和输出缓冲区
    std::vector<void*> buffers(2);
    size_t input_size = 3 * 640 * 640 * sizeof(float);
    cudaMalloc(&buffers[0], input_size);
    cudaMemcpy(buffers[0], resized_image.data, input_size, cudaMemcpyHostToDevice);

    // 获取输出维度
    nvinfer1::Dims output_dims = getBindingDimensionsCustom(1);
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    output_size *= sizeof(float);

    cudaMalloc(&buffers[1], output_size);

    // 执行推理
    context->executeV2(buffers.data());

    std::vector<float> output_data(output_size / sizeof(float));
    cudaMemcpy(output_data.data(), buffers[1], output_size, cudaMemcpyDeviceToHost);

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    std::vector<DetectionResult> detections;
    // 这里输出格式是 [x1, y1, x2, y2, score, class_id] 的形式，需要根据实际模型输出调整
    for (size_t i = 0; i < output_data.size(); i += 6) {
        float x1 = output_data[i] * image.cols / 640;
        float y1 = output_data[i + 1] * image.rows / 640;
        float x2 = output_data[i + 2] * image.cols / 640;
        float y2 = output_data[i + 3] * image.rows / 640;
        float score = output_data[i + 4];
        int class_id = static_cast<int>(output_data[i + 5]);
        cv::Mat box = (cv::Mat_<float>(4, 2) << x1, y1, x2, y1, x2, y2, x1, y2);
        detections.emplace_back(DetectionResult{box, score, class_id});
    }

    return detections;
}

// 保存检测结果到 txt 文件
void save_detection_result(const std::vector<DetectionResult>& detections, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    for (const auto& detection : detections) {
        const cv::Mat& box = detection.box;
        for (int i = 0; i < box.rows; ++i) {
            file << box.at<float>(i, 0) << " " << box.at<float>(i, 1) << " ";
        }
        file << detection.score << " " << detection.class_id << std::endl;
    }

    file.close();
}

// 读取 txt 文件中的四点数据、分数和类别 ID
std::vector<DetectionResult> read_detection_result(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return {};
    }

    std::vector<DetectionResult> detections;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        cv::Mat box = cv::Mat::zeros(4, 2, CV_32F);
        for (int i = 0; i < 4; ++i) {
            iss >> box.at<float>(i, 0) >> box.at<float>(i, 1);
        }
        float score;
        int class_id;
        iss >> score >> class_id;
        detections.emplace_back(DetectionResult{box, score, class_id});
    }

    file.close();
    return detections;
}

// 扩大检测框
cv::Mat expand_bounding_box(const cv::Mat& box, const cv::Mat& image, float expand_ratio = 0.1) {
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    for (int i = 0; i < box.rows; ++i) {
        x_coords.push_back(box.at<float>(i, 0));
        y_coords.push_back(box.at<float>(i, 1));
    }

    float x_min = *std::min_element(x_coords.begin(), x_coords.end());
    float x_max = *std::max_element(x_coords.begin(), x_coords.end());
    float y_min = *std::min_element(y_coords.begin(), y_coords.end());
    float y_max = *std::max_element(y_coords.begin(), y_coords.end());

    float width = x_max - x_min;
    float height = y_max - y_min;

    float expand_x = width * expand_ratio;
    float expand_y = height * expand_ratio;

    float new_x_min = std::max(0.0f, x_min - expand_x);
    float new_x_max = std::min((float)image.cols, x_max + expand_x);
    float new_y_min = std::max(0.0f, y_min - expand_y);
    float new_y_max = std::min((float)image.rows, y_max + expand_y);

    cv::Mat new_box = (cv::Mat_<float>(4, 2) << new_x_min, new_y_min, new_x_max, new_y_min, new_x_max, new_y_max, new_x_min, new_y_max);
    return new_box;
}

// 生成 ROI 区域
cv::Mat generate_roi(const cv::Mat& image, const cv::Mat& box) {
    int x1 = static_cast<int>(box.at<float>(0, 0));
    int y1 = static_cast<int>(box.at<float>(0, 1));
    int x2 = static_cast<int>(box.at<float>(2, 0));
    int y2 = static_cast<int>(box.at<float>(2, 1));
    return image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
}

// 解决装甲板灯条的精确定位问题，结合ROI处理
std::vector<DetectionResult> solve_light(const std::vector<DetectionResult>& armorVertices_vector, cv::Mat& img_raw, const std::string& engine_file_path) {
    std::vector<DetectionResult> p;
    std::string temp_image_path;
    for (const auto& armorVertices : armorVertices_vector) {
        if (armorVertices.box.rows == 4) {
            cv::Point2f up_left = cv::Point2f(armorVertices.box.at<float>(0, 0), armorVertices.box.at<float>(0, 1));
            cv::Point2f down_left = cv::Point2f(armorVertices.box.at<float>(1, 0), armorVertices.box.at<float>(1, 1));
            cv::Point2f down_right = cv::Point2f(armorVertices.box.at<float>(2, 0), armorVertices.box.at<float>(2, 1));
            cv::Point2f up_right = cv::Point2f(armorVertices.box.at<float>(3, 0), armorVertices.box.at<float>(3, 1));

            // roi区域
            cv::Size s = img_raw.size();
            float roi_width = abs((up_left.x - down_right.x) / 4);
            float roi_height = abs((up_left.y - down_right.y) / 4);

            cv::Rect roi;
            // 左边灯条
            down_left.x = std::max(down_left.x - roi_width, 0.0f);
            down_left.y = std::min(down_left.y + roi_height, static_cast<float>(s.height));
            up_left.x = std::max(up_left.x - roi_width, 0.0f);
            up_left.y = std::max(up_left.y - roi_height, 0.0f);
            roi.x = std::min(up_left.x, down_left.x);
            roi.y = std::min(up_left.y, down_left.y);
            up_right.x = std::min(up_right.x + roi_width, static_cast<float>(s.width));
            up_right.y = std::max(up_right.y - roi_height, 0.0f);
            down_right.x = std::min(down_right.x + roi_width, static_cast<float>(s.width));
            down_right.y = std::min(down_right.y + roi_height, static_cast<float>(s.height));
            roi.width = std::max(abs(up_left.x - down_right.x), abs(up_right.x - down_left.x));
            roi.height = std::max(abs(up_left.y - down_right.y), abs(up_right.y - down_left.y));

            cv::Mat clone_left = img_raw(roi).clone();
            temp_image_path = "temp_image.jpg";
            cv::imwrite(temp_image_path, clone_left);
            std::vector<DetectionResult> left_boxes = tensorrt_detection(temp_image_path, engine_file_path);
            std::remove(temp_image_path.c_str());

            for (auto& box : left_boxes) {
                box.box.at<float>(0, 0) += roi.x;
                box.box.at<float>(0, 1) += roi.y;
                box.box.at<float>(1, 0) += roi.x;
                box.box.at<float>(1, 1) += roi.y;
                box.box.at<float>(2, 0) += roi.x;
                box.box.at<float>(2, 1) += roi.y;
                box.box.at<float>(3, 0) += roi.x;
                box.box.at<float>(3, 1) += roi.y;
                p.push_back(box);
            }

            // 右边灯条
            down_left.x = std::max(down_right.x - roi_width, 0.0f);
            down_left.y = std::min(down_right.y + roi_height, static_cast<float>(s.height));
            up_left.x = std::max(up_right.x - roi_width, 0.0f);
            up_left.y = std::max(up_left.y - roi_height, 0.0f);
            roi.x = std::min(up_left.x, down_left.x);
            roi.y = std::min(up_left.y, down_left.y);
            up_right.x = std::min(up_right.x + roi_width, static_cast<float>(s.width));
            up_right.y = std::max(up_right.y - roi_height, 0.0f);
            down_right.x = std::min(down_right.x + roi_width, static_cast<float>(s.width));
            down_right.y = std::min(down_right.y + roi_height, static_cast<float>(s.height));
            roi.width = std::max(abs(up_left.x - down_right.x), abs(up_right.x - down_left.x));
            roi.height = std::max(abs(up_left.y - down_right.y), abs(up_right.y - down_left.y));

            cv::Mat clone_right = img_raw(roi).clone();
            temp_image_path = "temp_image.jpg";
            cv::imwrite(temp_image_path, clone_right);
            std::vector<DetectionResult> right_boxes = tensorrt_detection(temp_image_path, engine_file_path);
            std::remove(temp_image_path.c_str());

            for (auto& box : right_boxes) {
                box.box.at<float>(0, 0) += roi.x;
                box.box.at<float>(0, 1) += roi.y;
                box.box.at<float>(1, 0) += roi.x;
                box.box.at<float>(1, 1) += roi.y;
                box.box.at<float>(2, 0) += roi.x;
                box.box.at<float>(2, 1) += roi.y;
                box.box.at<float>(3, 0) += roi.x;
                box.box.at<float>(3, 1) += roi.y;
                p.push_back(box);
            }
        }
    }
    return p;
}

int main() {
    std::string image_folder = "/home/stark/桌面/ROI_Expand/images";
    std::string label_folder = "/home/stark/桌面/ROI_Expand/labels";
    std::string engine_file_path = "/home/stark/桌面/ROI_Expand/4pointsV16.engine";

    std::cout << "开始遍历图片文件夹: " << image_folder << std::endl;
    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")) {
            std::string image_path = entry.path().string();
            std::string image_name = entry.path().stem().string();
            std::string label_path = label_folder + "/" + image_name + ".txt";

            std::cout << "正在处理图片: " << image_path << std::endl;
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Failed to read image: " << image_path << std::endl;
                continue;
            }

            std::cout << "正在读取标签文件: " << label_path << std::endl;
            // 读取标签文件
            std::vector<DetectionResult> read_boxes = read_detection_result(label_path);

            // 解决装甲板灯条的精确定位问题，结合ROI处理
            std::cout << "正在处理灯条定位问题..." << std::endl;
            std::vector<DetectionResult> refined_boxes = solve_light(read_boxes, image, engine_file_path);

            std::vector<DetectionResult> expanded_boxes;
            for (const auto& box : refined_boxes) {
                // 扩大检测框
                cv::Mat expanded_box = expand_bounding_box(box.box, image);
                expanded_boxes.emplace_back(DetectionResult{expanded_box, box.score, box.class_id});

                // 生成 ROI 区域
                cv::Mat roi = generate_roi(image, expanded_box);

                // 绘制原始检测框（绿色）
                cv::rectangle(image, cv::Point(box.box.at<float>(0, 0), box.box.at<float>(0, 1)), cv::Point(box.box.at<float>(2, 0), box.box.at<float>(2, 1)), cv::Scalar(0, 255, 0), 2);

                // 绘制扩展后的检测框（红色）
                cv::rectangle(image, cv::Point(expanded_box.at<float>(0, 0), expanded_box.at<float>(0, 1)), cv::Point(expanded_box.at<float>(2, 0), expanded_box.at<float>(2, 1)), cv::Scalar(0, 0, 255), 2);

                // 显示原始图片和 ROI 区域
                cv::imshow("Original Image with ROI", image);
                cv::imshow("ROI", roi);
                cv::waitKey(0);
            }

            // 保存扩大后的检测框结果到新的 txt 文件
            std::string output_label_path = label_folder + "/" + image_name + "_expanded.txt";
            std::cout << "正在保存处理后的标签文件: " << output_label_path << std::endl;
            save_detection_result(expanded_boxes, output_label_path);
        }
    }
    cv::waitKey(1000);//此处待调试
    cv::destroyAllWindows();
    return 0;
}    

