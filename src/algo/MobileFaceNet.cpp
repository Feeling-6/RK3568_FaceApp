#include "MobileFaceNet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MobileFaceNet::MobileFaceNet() : ctx(0), is_init(false) {}

MobileFaceNet::~MobileFaceNet() {
    release();
}

int MobileFaceNet::init(const std::string& model_path) {
    int model_len = 0;
    unsigned char* model_data = load_model(model_path.c_str(), &model_len);
    if (!model_data) return -1;

    // 1. 初始化 RKNN 上下文
    int ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 2. 查询输入输出数量
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    
    // 3. 获取输入输出属性
    input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
    memset(input_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
    memset(output_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    is_init = true;
    return 0;
}

int MobileFaceNet::extractFeature(const cv::Mat& face_img, std::vector<float>& feature) {
    if (!is_init) return -1;

    // 1. 设置输入数据
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8; // 常用格式
    inputs[0].size = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
    inputs[0].fmt = RKNN_TENSOR_NHWC; // RKNN 默认通常是 NHWC
    inputs[0].buf = face_img.data;

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    // 2. 执行推理
    rknn_run(ctx, NULL);

    // 3. 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // 让 SDK 自动做反量化（dequantization）

    int ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) return -1;

    // 4. 拷贝特征值 (w600k_mbf 通常输出 128 或 512 维)
    int count = output_attrs[0].n_elems;
    float* out_data = (float*)outputs[0].buf;
    feature.assign(out_data, out_data + count);

    // L2 归一化（人脸识别比对前通常需要归一化）
    cv::Mat feat_mat(1, count, CV_32FC1, feature.data());
    cv::normalize(feat_mat, feat_mat, 1.0, 0, cv::NORM_L2);

    rknn_outputs_release(ctx, io_num.n_output, outputs);
    return 0;
}

void MobileFaceNet::release() {
    if (ctx > 0) rknn_destroy(ctx);
    if (input_attrs) free(input_attrs);
    if (output_attrs) free(output_attrs);
}

unsigned char* MobileFaceNet::load_model(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) return NULL;
    fseek(fp, 0L, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}