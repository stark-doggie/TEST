#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>

// �Զ���� Logger �࣬���ڴ��� TensorRT ����־��Ϣ
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

// ��ȡ .engine �ļ�
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

// ʹ�� TensorRT ģ�ͽ��м��
std::vector<cv::Mat> tensorrt_detection(const std::string& image_path, const std::string& engine_file_path) {
    Logger logger;
    // ���� TensorRT ����ʱ����
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return {};
    }

    // ��ȡ .engine �ļ�
    std::vector<char> engine_data = read_engine_file(engine_file_path);
    if (engine_data.empty()) {
        return {};
    }

    // �� .engine �ļ������з����л����� TensorRT ����
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));
    if (!engine) {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        return {};
    }

    // ���� TensorRT �����Ķ���
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create TensorRT execution context." << std::endl;
        return {};
    }

    // ��ȡ����ͼ��
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return {};
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // ׼����������������
    std::vector<void*> buffers(2);
    size_t input_size = 3 * 640 * 640 * sizeof(float);
    cudaMalloc(&buffers[0], input_size);
    cudaMemcpy(buffers[0], resized_image.data, input_size, cudaMemcpyHostToDevice);

    size_t output_size = engine->getBindingDimensions(1).d[0] * sizeof(float);
    cudaMalloc(&buffers[1], output_size);

    // ִ������
    context->executeV2(buffers.data());

    std::vector<float> output_data(engine->getBindingDimensions(1).d[0]);
    cudaMemcpy(output_data.data(), buffers[1], output_size, cudaMemcpyDeviceToHost);

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    std::vector<cv::Mat> boxes;
    // ������������ʽ�� [x1, y1, x2, y2, score, class_id] ����ʽ����Ҫ����ʵ��ģ���������
    for (size_t i = 0; i < output_data.size(); i += 6) {
        float x1 = output_data[i] * image.cols / 640;
        float y1 = output_data[i + 1] * image.rows / 640;
        float x2 = output_data[i + 2] * image.cols / 640;
        float y2 = output_data[i + 3] * image.rows / 640;
        cv::Mat box = (cv::Mat_<float>(4, 2) << x1, y1, x2, y1, x2, y2, x1, y2);
        boxes.push_back(box);
    }

    return boxes;
}

// ���������� txt �ļ�
void save_detection_result(const std::vector<cv::Mat>& boxes, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    for (const auto& box : boxes) {
        for (int i = 0; i < box.rows; ++i) {
            file << box.at<float>(i, 0) << " " << box.at<float>(i, 1) << " ";
        }
        file << std::endl;
    }

    file.close();
}

// ��ȡ txt �ļ��е��ĵ�����
std::vector<cv::Mat> read_detection_result(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return {};
    }

    std::vector<cv::Mat> boxes;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        cv::Mat box = cv::Mat::zeros(4, 2, CV_32F);
        for (int i = 0; i < 4; ++i) {
            iss >> box.at<float>(i, 0) >> box.at<float>(i, 1);
        }
        boxes.push_back(box);
    }

    file.close();
    return boxes;
}

// �������
cv::Mat expand_bounding_box(const cv::Mat& box, const cv::Mat& image, float expand_ratio = 0.1) {
    float x_min = cv::min(box.col(0))[0];
    float x_max = cv::max(box.col(0))[0];
    float y_min = cv::min(box.col(1))[0];
    float y_max = cv::max(box.col(1))[0];

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

// ���� ROI ����
cv::Mat generate_roi(const cv::Mat& image, const cv::Mat& box) {
    int x1 = static_cast<int>(box.at<float>(0, 0));
    int y1 = static_cast<int>(box.at<float>(0, 1));
    int x2 = static_cast<int>(box.at<float>(2, 0));
    int y2 = static_cast<int>(box.at<float>(2, 1));
    return image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
}

int main() {
    // ͼƬ·��
    std::string image_path = "Armor.jpg";
    // .engine ģ���ļ�·��
    std::string engine_file_path = "4pointsV16.engine";

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    // ʹ�� TensorRT ģ�ͽ��м��
    std::vector<cv::Mat> detected_boxes = tensorrt_detection(image_path, engine_file_path);

    // ���������� txt �ļ�
    save_detection_result(detected_boxes, "detection_result.txt");

    // ��ȡ txt �ļ��е��ĵ�����
    std::vector<cv::Mat> read_boxes = read_detection_result("detection_result.txt");

    for (const auto& box : read_boxes) {
        // �������
        cv::Mat expanded_box = expand_bounding_box(box, image);

        // ���� ROI ����
        cv::Mat roi = generate_roi(image, expanded_box);

        // ����ԭʼ������ɫ��
        cv::rectangle(image, cv::Point(box.at<float>(0, 0), box.at<float>(0, 1)), cv::Point(box.at<float>(2, 0), box.at<float>(2, 1)), cv::Scalar(0, 255, 0), 2);

        // ������չ��ļ��򣨺�ɫ��
        cv::rectangle(image, cv::Point(expanded_box.at<float>(0, 0), expanded_box.at<float>(0, 1)), cv::Point(expanded_box.at<float>(2, 0), expanded_box.at<float>(2, 1)), cv::Scalar(0, 0, 255), 2);

        // ��ʾԭʼͼƬ�� ROI ����
        cv::imshow("Original Image with ROI", image);
        cv::imshow("ROI", roi);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}