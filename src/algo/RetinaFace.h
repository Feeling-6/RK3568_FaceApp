#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "im2d.hpp"
#include "rga.h"
#include "device/CameraManager.h"

// 定义检测结果结构体
struct FaceInfo {
    cv::Rect box;           // 人脸框
    float score;            // 置信度
    cv::Point2f landmarks[5]; // 5个关键点 (左眼, 右眼, 鼻, 左嘴, 右嘴)
};

class RetinaFace {
public:
    RetinaFace(const std::string& modelPath);
    ~RetinaFace();

    // 初始化模型（加载RKNN等）
    int init();

    // 核心业务函数：从相机获取帧 -> 检测 -> 对齐 -> 返回112x112人脸
    // 如果没检测到人脸，返回空的 cv::Mat
    cv::Mat getAlignedFaceFromCamera(CameraManager& camera);

private:
    // RKNN 上下文
    rknn_context ctx;
    unsigned char* model_data;
    int model_data_size;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;

    // 模型输入尺寸
    const int MODEL_WIDTH = 320;
    const int MODEL_HEIGHT = 320;

    // 存储预计算的 Anchor (cx, cy, w, h)
    // 对应官方代码里的 BOX_PRIORS_320
    std::vector<std::vector<float>> priors; 

    int detect(const cv::Mat& inputImg, std::vector<FaceInfo>& faces);
    cv::Mat preprocessFace(const cv::Mat& img, const cv::Point2f landmarks[5]);
    
    // 初始化 Anchors
    void initPriors();
};

#endif // RETINAFACE_H
