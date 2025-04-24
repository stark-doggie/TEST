#include "detector_dl/Detector.hpp"

Detector::Detector(const int& NUM_CLASSES, const std::string& TARGET_COLOUR, const float& NMS_THRESH, const float& BBOX_CONF_THRESH,
    const int& INPUT_W, const int& INPUT_H, const int& CKPT_NUM, const std::string& engine_file_path, const ArmorParams& a)
    : NUM_CLASSES(NUM_CLASSES), TARGET_COLOUR(TARGET_COLOUR), NMS_THRESH(NMS_THRESH), BBOX_CONF_THRESH(BBOX_CONF_THRESH),
    INPUT_W(INPUT_W), INPUT_H(INPUT_H), CKPT_NUM(CKPT_NUM), engine_file_path(engine_file_path), a(a) {

    std::cout << "进入Detector构造函数" << std::endl;
    InitModelEngine();
    std::cout << "模型引擎初始化完成" << std::endl;
    AllocMem();
    std::cout << "内存分配完成" << std::endl;
}


void Detector::InitModelEngine()
{
    std::cout << "进入InitModelEngine函数" << std::endl;
    cudaSetDevice(DEVICE);
    std::ifstream file{ this->engine_file_path, std::ios::binary };
    size_t size{ 0 };
    char* trtModelStreamDet{ nullptr };
    if (file.good())
    {
        std::cout << "引擎文件状态良好，开始读取文件大小" << std::endl;
        file.seekg(0, file.end);
        size = file.tellg();
        std::cout << "获取到文件大小，将文件指针移到开头" << std::endl;
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        std::cout << "已分配内存用于trtModelStreamDet，开始读取文件内容" << std::endl;
        file.read(trtModelStreamDet, size);
        file.close();
        std::cout << "文件内容读取完毕，文件已关闭" << std::endl;
    }
    std::cout << "开始创建TensorRT运行时" << std::endl;
    runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    std::cout << "TensorRT运行时创建成功，开始反序列化CUDA引擎" << std::endl;
    engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr);
    std::cout << "CUDA引擎反序列化成功，开始创建执行上下文" << std::endl;
    this->context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    std::cout << "执行上下文创建成功，释放trtModelStreamDet内存" << std::endl;
    delete[] trtModelStreamDet;
    std::cout << "trtModelStreamDet内存已释放，离开InitModelEngine函数" << std::endl;
}

void Detector::AllocMem()
{
    std::cout << "进入AllocMem函数" << std::endl;
    inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    OUTPUT_CANDIDATES = out_dims.d[1];
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        output_size *= out_dims.d[i];
    }

    std::cout << "开始分配输入缓冲区内存" << std::endl;
    CHECK(cudaMalloc(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    std::cout << "输入缓冲区内存分配完成，开始分配输出缓冲区内存" << std::endl;
    CHECK(cudaMalloc(&buffers[outputIndex], sizeof(float) * output_size));

    std::cout << "开始分配img_host主机内存" << std::endl;
    CHECK(cudaMallocHost(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    std::cout << "img_host主机内存分配完成，开始分配img_device设备内存" << std::endl;
    CHECK(cudaMalloc(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    std::cout << "开始分配affine_matrix_d2i_host主机内存" << std::endl;
    CHECK(cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6));
    std::cout << "affine_matrix_d2i_host主机内存分配完成，开始分配affine_matrix_d2i_device设备内存" << std::endl;
    CHECK(cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6));

    std::cout << "开始分配decode_ptr_device设备内存" << std::endl;
    CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
    std::cout << "decode_ptr_device设备内存分配完成，离开AllocMem函数" << std::endl;
}


Target Detector::detect(cv::Mat& frame, bool show_img = false)
{
    std::cout << "进入detect函数" << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 预处理
    AffineMatrix afmt;
    std::cout << "调用getd2i函数进行预处理设置" << std::endl;
    getd2i(afmt, { INPUT_W, INPUT_H }, cv::Size(frame.cols, frame.rows));
    float* buffer_idx = (float*)buffers[inputIndex];
    size_t img_size = frame.cols * frame.rows * 3;

    std::cout << "将仿射矩阵数据复制到主机" << std::endl;
    memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
    std::cout << "将仿射矩阵数据异步复制到设备" << std::endl;
    CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));

    std::cout << "将图像数据复制到img_host" << std::endl;
    memcpy(img_host, frame.data, img_size);
    std::cout << "将图像数据异步复制到img_device" << std::endl;
    CHECK(cudaMemcpyAsync(img_device, img_host, img_size, cudaMemcpyHostToDevice, stream));
    std::cout << "调用preprocess_kernel_img函数进行图像预处理" << std::endl;
    preprocess_kernel_img(img_device, frame.cols, frame.rows,
        buffer_idx, INPUT_W, INPUT_H,
        affine_matrix_d2i_device, stream);
    // 推理
    std::cout << "开始推理，调用enqueueV2函数" << std::endl;
    (*context_det).enqueueV2((void**)buffers, stream, nullptr);
    float* predict = (float*)buffers[outputIndex];
    // 后处理
    std::cout << "开始后处理，重置decode_ptr_device内存" << std::endl;
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
    std::cout << "调用decode_kernel_invoker函数进行解码" << std::endl;
    decode_kernel_invoker(
        predict, NUM_BOX_ELEMENT, OUTPUT_CANDIDATES, NUM_CLASSES,
        CKPT_NUM, BBOX_CONF_THRESH, affine_matrix_d2i_device,
        decode_ptr_device, MAX_OBJECTS, stream);
    std::cout << "调用nms_kernel_invoker函数进行非极大值抑制" << std::endl;
    nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream, NUM_BOX_ELEMENT);
    std::cout << "将decode_ptr_device数据异步复制回主机" << std::endl;
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
    std::cout << "等待CUDA流同步" << std::endl;
    cudaStreamSynchronize(stream);

    color_list = create_color_list(CKPT_NUM);
    std::vector<Outpost> outposts;
    std::vector<bbox> boxes;
    int boxes_count = 0;
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    bool detected = true;
    this->is_detected = (count != 0) ? true : false;

    if (this->is_detected)
    {
        if (CKPT_NUM == 4)
        {
            std::cout << "检测到目标，开始处理检测结果（CKPT_NUM == 4）" << std::endl;
            for (int i = 0; i < count; ++i)
            {
                int basic_pos = 1 + i * NUM_BOX_ELEMENT;
                int keep_flag = decode_ptr_host[basic_pos + 6];
                if (keep_flag == 1)
                {
                    boxes_count += 1;
                    bbox box;
                    box.x1 = decode_ptr_host[basic_pos + 0];
                    box.y1 = decode_ptr_host[basic_pos + 1];
                    box.x2 = decode_ptr_host[basic_pos + 2];
                    box.y2 = decode_ptr_host[basic_pos + 3];
                    box.score = decode_ptr_host[basic_pos + 4];
                    box.class_id = decode_ptr_host[basic_pos + 5];
                    // 输出class_id
                    std::cout << "检测到目标，class_id: " << box.class_id << std::endl;

                    int landmark_pos = basic_pos + 7;
                    for (int id = 0; id < CKPT_NUM; id += 1)
                    {
                        box.landmarks[2 * id] = decode_ptr_host[landmark_pos + 2 * id];
                        box.landmarks[2 * id + 1] = decode_ptr_host[landmark_pos + 2 * id + 1];
                    }
                    boxes.push_back(box);
                }
            }
        }

        if (CKPT_NUM == 4)
        {
            detected = false;
            std::cout << "根据目标颜色和类别筛选前哨站（CKPT_NUM == 4）" << std::endl;
            for (auto box : boxes)
            {
                // 输出当前处理目标的class_id
                std::cout << "正在处理目标，class_id: " << box.class_id << std::endl;
                if (this->TARGET_COLOUR == "GREEN" && box.class_id == 6)
                {
                    std::vector<cv::Point2f> points{ cv::Point2f(box.landmarks[0], box.landmarks[1]),
                                                     cv::Point2f(box.landmarks[2], box.landmarks[3]),
                                                     cv::Point2f(box.landmarks[4], box.landmarks[5]),
                                                     cv::Point2f(box.landmarks[6], box.landmarks[7]) };
                    Outpost outpost = Outpost(points);

                    if (outpost.if_value)
                    {
                        detected = true;
                        outposts.emplace_back(outpost);
                    }
                }
            }
        }

        if (show_img && detected)
        {
            std::cout << "检测到前哨站，绘制检测结果" << std::endl;
            for (int i = 0; i < outposts.size(); i++)
            {
                for (int j = 0; j < CKPT_NUM; j++)
                {
                    cv::Scalar color = cv::Scalar(color_list[j][0], color_list[j][1], color_list[j][2]);
                    for (const auto& vertex : outposts[i].outpostVertices_vector) {
                        cv::circle(frame, vertex, 2, color, -1);
                    }
                }
                std::string label = "Outpost";
                cv::putText(frame, label, cv::Point(outposts[i].center.x, outposts[i].center.y - 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);

                // 绘制前哨站的边
                for (size_t j = 0; j < outposts[i].outpostVertices_vector.size(); ++j) {
                    cv::line(frame,
                        outposts[i].outpostVertices_vector[j],
                        outposts[i].outpostVertices_vector[(j + 1) % outposts[i].outpostVertices_vector.size()],
                        cv::Scalar(0, 255, 0),
                        3);
                }
            }
        }
        else
        {
            std::cout << "未检测到前哨站，绘制提示信息" << std::endl;
            cv::putText(frame, "No Detected!", cv::Point2f{ 100, 100 }, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
        }

    }
    std::cout << "销毁CUDA流" << std::endl;
    cudaStreamDestroy(stream);

    if (CKPT_NUM == 4)
        return outposts;
}

void Detector::Release()
{
    std::cout << "进入Release函数，开始销毁相关对象和释放内存" << std::endl;
    context_det->destroy();
    std::cout << "已销毁IExecutionContext对象" << std::endl;
    engine_det->destroy();
    std::cout << "已销毁ICudaEngine对象" << std::endl;
    runtime_det->destroy();
    std::cout << "已销毁IRuntime对象" << std::endl;

    delete_color_list(color_list, CKPT_NUM);
    std::cout << "已调用delete_color_list函数来释放颜色列表内存" << std::endl;
    cudaStreamDestroy(stream);
    std::cout << "已销毁CUDA流" << std::endl;
    CHECK(cudaFree(affine_matrix_d2i_device));
    std::cout << "已释放affine_matrix_d2i_device设备内存" << std::endl;
    CHECK(cudaFreeHost(affine_matrix_d2i_host));
    std::cout << "已释放affine_matrix_d2i_host主机内存" << std::endl;
    CHECK(cudaFree(img_device));
    std::cout << "已释放img_device设备内存" << std::endl;
    CHECK(cudaFreeHost(img_host));
    std::cout << "已释放img_host主机内存" << std::endl;
    CHECK(cudaFree(buffers[inputIndex]));
    std::cout << "已释放buffers[inputIndex]设备内存" << std::endl;
    CHECK(cudaFree(buffers[outputIndex]));
    std::cout << "已释放buffers[outputIndex]设备内存" << std::endl;
    CHECK(cudaFree(decode_ptr_device));
    std::cout << "已释放decode_ptr_device设备内存" << std::endl;
    delete[] decode_ptr_host;
    std::cout << "已释放decode_ptr_host主机内存" << std::endl;
    //decode_ptr_host = nullptr;
    std::cout << "Release函数执行完毕，相关对象已销毁，内存已释放" << std::endl;
}

float** Detector::create_color_list(int num_key_points) {
    float** color_list = new float*[num_key_points];
    for (int i = 0; i < num_key_points; ++i) {
        color_list[i] = new float[3];
        // Assign some default colors here or adjust according to your preference
        // For simplicity, I'll assign red, green, blue colors alternatively
        color_list[i][0] = (i % 3 == 0) ? 255 : 0;  // Red
        color_list[i][1] = (i % 3 == 1) ? 255 : 0;  // Green
        color_list[i][2] = (i % 3 == 2) ? 255 : 0;  // Blue
    }
    return color_list;
}

// Function to deallocate memory for color list
void Detector::delete_color_list(float** color_list, int num_key_points) {
    for (int i = 0; i < num_key_points; ++i) {
        delete[] color_list[i];
    }
    delete[] color_list;
}

/*
#include "detector_dl/Detector.hpp"

Detector::Detector(const int& NUM_CLASSES, const std::string& TARGET_COLOUR, const float& NMS_THRESH, const float& BBOX_CONF_THRESH,
                   const int& INPUT_W, const int& INPUT_H, const int& CKPT_NUM, const std::string& engine_file_path, const ArmorParams& a)
    : NUM_CLASSES(NUM_CLASSES), TARGET_COLOUR(TARGET_COLOUR), NMS_THRESH(NMS_THRESH), BBOX_CONF_THRESH(BBOX_CONF_THRESH),
      INPUT_W(INPUT_W), INPUT_H(INPUT_H), CKPT_NUM(CKPT_NUM), engine_file_path(engine_file_path), a(a) {

    InitModelEngine();
    AllocMem();

}


void Detector::InitModelEngine()
{
    cudaSetDevice(DEVICE);
    std::ifstream file {this->engine_file_path, std::ios::binary};
    size_t size {0};
    char *trtModelStreamDet {nullptr};
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
    }
    runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr);
    this->context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    delete[] trtModelStreamDet;
}

void Detector::AllocMem()
{
    inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    OUTPUT_CANDIDATES = out_dims.d[1];
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        output_size *= out_dims.d[i];
    }

    // 尝试优化:
    CHECK(cudaMalloc(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    CHECK(cudaMalloc(&buffers[outputIndex], sizeof(float) * output_size));
    // CHECK(cudaMallocManaged(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    // CHECK(cudaMallocManaged(&buffers[outputIndex], sizeof(float) * output_size));

    CHECK(cudaMallocHost(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    //---------------------------------------------------------------------------
    // CHECK(cudaMallocManaged(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3, cudaMemAttachHost));
    // CHECK(cudaMallocManaged(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));  

    CHECK(cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6));
    CHECK(cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6));
    //---------------------------------------------------------------------------
    // CHECK(cudaMallocManaged(&affine_matrix_d2i_device, sizeof(float) * 6));


    CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
    // CHECK(cudaMallocManaged(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
}


Target Detector::detect(cv::Mat &frame, bool show_img = false)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 预处理
    AffineMatrix afmt;
    // CHECK(cudaMallocManaged(&(afmt.d2i), sizeof(float) * 6, cudaMemAttachHost));
    getd2i(afmt, {INPUT_W, INPUT_H}, cv::Size(frame.cols, frame.rows)); // TODO
    float *buffer_idx = (float *)buffers[inputIndex];
    size_t img_size = frame.cols * frame.rows * 3;

    memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
    CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaStreamAttachMemAsync(stream, afmt.d2i, 0, cudaMemAttachGlobal));
    // CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));

    memcpy(img_host, frame.data, img_size);
    CHECK(cudaMemcpyAsync(img_device, img_host, img_size, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, frame.cols, frame.rows, 
        buffer_idx, INPUT_W, INPUT_H, 
        affine_matrix_d2i_device, stream);
    // 推理
    (*context_det).enqueueV2((void **)buffers, stream, nullptr);
    float *predict = (float *)buffers[outputIndex];
    // 后处理
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
    decode_kernel_invoker(
        predict, NUM_BOX_ELEMENT, OUTPUT_CANDIDATES, NUM_CLASSES, 
        CKPT_NUM, BBOX_CONF_THRESH, affine_matrix_d2i_device, 
        decode_ptr_device, MAX_OBJECTS, stream);
    nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream, NUM_BOX_ELEMENT);    
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaStreamAttachMemAsync(stream, decode_ptr_device, 0, cudaMemAttachHost));
    cudaStreamSynchronize(stream);

    color_list = create_color_list(CKPT_NUM);
    std::vector<Armor> armors;
    std::vector<bbox> boxes;
    int boxes_count = 0;
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    bool detected = true;
    this->is_detected = (count !=0) ? true : false;
    //bool show_R = false;

    if (this->is_detected)
    {
        if(CKPT_NUM == 4)
        {
            for (int i = 0; i < count; ++i)
            {
                int basic_pos = 1 + i * NUM_BOX_ELEMENT;
                int keep_flag = decode_ptr_host[basic_pos + 6];
                if (keep_flag == 1)
                {
                    boxes_count += 1;
                    bbox box;
                    box.x1 = decode_ptr_host[basic_pos + 0];
                    box.y1 = decode_ptr_host[basic_pos + 1];
                    box.x2 = decode_ptr_host[basic_pos + 2];
                    box.y2 = decode_ptr_host[basic_pos + 3];
                    box.score = decode_ptr_host[basic_pos + 4];
                    box.class_id = decode_ptr_host[basic_pos + 5];

                    int landmark_pos = basic_pos + 7;
                    for(int id = 0; id < CKPT_NUM; id += 1)
                    {
                        box.landmarks[2 * id] = decode_ptr_host[landmark_pos + 2 * id];
                        box.landmarks[2 * id + 1] = decode_ptr_host[landmark_pos + 2 * id + 1];
                    }
                    boxes.push_back(box);
                }
            }
        }

        if(CKPT_NUM == 4)
        {
            detected = false;
            for (auto box : boxes)
            {
                // bl->tl->tr->br 左下 左上 右上 右下
                std::vector<cv::Point2f> points{cv::Point2f(box.landmarks[0], box.landmarks[1]),
                                                cv::Point2f(box.landmarks[2], box.landmarks[3]),
                                                cv::Point2f(box.landmarks[4], box.landmarks[5]),
                                                cv::Point2f(box.landmarks[6], box.landmarks[7])};
                Armor armor = Armor{points};

                float light_left_length = abs(box.landmarks[1] - box.landmarks[3]);
                float light_right_length = abs(box.landmarks[7] - box.landmarks[5]);
                float avg_light_length = (light_left_length + light_right_length) / 2;
                cv::Point2f light_left_center = cv::Point2f((box.landmarks[6] + box.landmarks[0]) / 2, (box.landmarks[7] + box.landmarks[1]) / 2);
                cv::Point2f light_right_center = cv::Point2f((box.landmarks[2] + box.landmarks[4]) / 2, (box.landmarks[3] + box.landmarks[5]) / 2);
                float center_distance = cv::norm(light_left_center - light_right_center) / avg_light_length;

                std::cout<< "TARGET_COLOUR: " << this->TARGET_COLOUR <<std::endl;
                std::cout<< "box.class_id: " << box.class_id <<std::endl;
                std::cout<< "CKPT_NUM: " << CKPT_NUM <<std::endl;
                std::cout<< "NUM_CLASSES: " << NUM_CLASSES <<std::endl;

                if(this->TARGET_COLOUR == "BLUE" && box.class_id <=8 && box.class_id >=0)
                {
                    armor.type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
                    if(box.class_id == 0)
                        armor.number = "guard";
                    else if(box.class_id >= 1 && box.class_id <= 5)
                        armor.number = std::to_string(box.class_id);
                    else if(box.class_id == 6)
                    {
                        armor.number = "outpost";
                    }
                    else if(box.class_id == 7||box.class_id == 8){
                        armor.number = "base";
                    }

                    detected = true;
                    armors.emplace_back(armor);

                }
                else if(this->TARGET_COLOUR == "RED" && box.class_id <=17 && box.class_id >=9 )
                {
                    armor.type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
                    if(box.class_id == 9)
                        armor.number = "guard";
                    else if(box.class_id >= 10 && box.class_id <= 14)
                        armor.number = std::to_string(box.class_id);
                    else if(box.class_id == 15)
                    {
                        armor.number = "outpost";
                    }
                    else if(box.class_id == 16||box.class_id == 17)
                        armor.number = "base";

                    detected = true;
                    armors.emplace_back(armor);
                }

                std::cout<<"armor: "<<armor.number<<std::endl;
            }
        }

        if (show_img && detected)
        {
            std::cout << "Detected Armor!" << std::endl;
            for (int i = 0; i<boxes_count; i++)
            {
                if (CKPT_NUM == 4 && ((this->TARGET_COLOUR == "BLUE" && boxes[i].class_id <=8 && boxes[i].class_id >=0) ||  (this->TARGET_COLOUR == "RED" && boxes[i].class_id <=17 && boxes[i].class_id >=9)))
                {
                    for (int j= 0; j<CKPT_NUM; j++)
                    {
                        cv::Scalar color = cv::Scalar(color_list[j][0], color_list[j][1], color_list[j][2]);
                        cv::circle(frame,cv::Point(boxes[i].landmarks[2*j], boxes[i].landmarks[2*j+1]), 2, color, -1);
                    }
                    std::string label = std::to_string(boxes[i].class_id);
                    cv::putText(frame, label, cv::Point(boxes[i].x1, boxes[i].y1 - 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);

                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[0], boxes[i].landmarks[1]),
                            cv::Point(boxes[i].landmarks[4], boxes[i].landmarks[5]),
                            cv::Scalar(0,255,0),
                            3);
                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[2], boxes[i].landmarks[3]),
                            cv::Point(boxes[i].landmarks[6], boxes[i].landmarks[7]),
                            cv::Scalar(0,255,0),
                            3);
                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[0], boxes[i].landmarks[1]),
                            cv::Point(boxes[i].landmarks[2], boxes[i].landmarks[3]),
                            cv::Scalar(0,255,0),
                            3);
                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[4], boxes[i].landmarks[5]),
                            cv::Point(boxes[i].landmarks[6], boxes[i].landmarks[7]),
                            cv::Scalar(0,255,0),
                            3);
                }
            }
        }
        else
        {
            cv::putText(frame, "No Detected!", cv::Point2f{100, 100}, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
        }

    }
    cudaStreamDestroy(stream);

    if(CKPT_NUM == 4)
        return armors;
}

void Detector::Release()
{
    context_det->destroy();
    engine_det->destroy();
    runtime_det->destroy();

    delete_color_list(color_list, CKPT_NUM);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(affine_matrix_d2i_device));
    CHECK(cudaFreeHost(affine_matrix_d2i_host));
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    //decode_ptr_host = nullptr;
}

float** Detector::create_color_list(int num_key_points) {
    float** color_list = new float*[num_key_points];
    for (int i = 0; i < num_key_points; ++i) {
        color_list[i] = new float[3];
        // Assign some default colors here or adjust according to your preference
        // For simplicity, I'll assign red, green, blue colors alternatively
        color_list[i][0] = (i % 3 == 0) ? 255 : 0;  // Red
        color_list[i][1] = (i % 3 == 1) ? 255 : 0;  // Green
        color_list[i][2] = (i % 3 == 2) ? 255 : 0;  // Blue
    }
    return color_list;
}

// Function to deallocate memory for color list
void Detector::delete_color_list(float** color_list, int num_key_points) {
    for (int i = 0; i < num_key_points; ++i) {
        delete[] color_list[i];
    }
    delete[] color_list;
}
*/