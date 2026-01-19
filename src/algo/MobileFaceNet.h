#ifndef MOBILEFACENET_H
#define MOBILEFACENET_H

#include <vector>
#include <string>
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

class MobileFaceNet {
public:
    MobileFaceNet();
    ~MobileFaceNet();

    // 加载模型
    int init(const std::string& model_path);
    
    // 执行推理：输入112x112的Mat，输出特征向量
    int extractFeature(const cv::Mat& face_img, std::vector<float>& feature);

private:
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    
    bool is_init = false;
    const int IMG_WIDTH = 112;
    const int IMG_HEIGHT = 112;
    const int IMG_CHANNELS = 3;

    unsigned char* load_model(const char* filename, int* model_size);
    void release();
};

#endif