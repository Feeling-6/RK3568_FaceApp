#include "RetinaFace.h"
#include <fstream>
#include <iostream>
#include <algorithm>

// 112x112 参考点
static float REFERENCE_PTS_112[5][2] = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f}
};

static inline float clip(float x, float min, float max) {
    return x < min ? min : (x > max ? max : x);
}

RetinaFace::RetinaFace(const std::string& modelPath) : ctx(0), model_data(nullptr) {
    std::ifstream ifs(modelPath, std::ios::binary);
    if (ifs.is_open()) {
        ifs.seekg(0, std::ios::end);
        model_data_size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        model_data = new unsigned char[model_data_size];
        ifs.read((char*)model_data, model_data_size);
        ifs.close();
    } else {
        std::cerr << "[RetinaFace] Error loading model: " << modelPath << std::endl;
    }
}

RetinaFace::~RetinaFace() {
    if (ctx) rknn_destroy(ctx);
    if (model_data) delete[] model_data;
    if (input_attrs) delete[] input_attrs;
    if (output_attrs) delete[] output_attrs;
}

// ---------------------------------------------------------
// 动态生成 Anchors
// ---------------------------------------------------------
void RetinaFace::initPriors() {
    // 针对 320x320 的配置
    // Strides: 8, 16, 32
    // MinSizes: [16,32], [64,128], [256,512]
    int strides[] = {8, 16, 32};
    std::vector<std::vector<float>> min_sizes = {
        {16.0f, 32.0f}, {64.0f, 128.0f}, {256.0f, 512.0f}
    };
    
    priors.clear();
    for (int k = 0; k < 3; k++) {
        int stride = strides[k];
        int feature_w = (int)ceil(MODEL_WIDTH / (float)stride);
        int feature_h = (int)ceil(MODEL_HEIGHT / (float)stride);
        
        for (int i = 0; i < feature_h; i++) {
            for (int j = 0; j < feature_w; j++) {
                for (auto min_size : min_sizes[k]) {
                    float s_kx = min_size / MODEL_WIDTH;
                    float s_ky = min_size / MODEL_HEIGHT;
                    float cx = (j + 0.5f) * stride / MODEL_WIDTH;
                    float cy = (i + 0.5f) * stride / MODEL_HEIGHT;
                    priors.push_back({cx, cy, s_kx, s_ky});
                }
            }
        }
    }
    // 320x320 应该生成 4200 个 anchor
    // std::cout << "Priors generated: " << priors.size() << std::endl;
}

int RetinaFace::init() {
    if (!model_data) return -1;
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) return -1;

    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    output_attrs = new rknn_tensor_attr[io_num.n_output];

    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }
    
    // 生成 Anchor
    initPriors();
    
    return 0;
}

// ---------------------------------------------------------
// 推理核心 (已适配官方 Output 格式)
// ---------------------------------------------------------
int RetinaFace::detect(const cv::Mat& inputImg, std::vector<FaceInfo>& faces) {
    if(!ctx) return -1;

    // 准备目标容器 (320x320, RGB)
    // 注意：RKNN通常需要RGB格式，而OpenCV默认是BGR
    cv::Mat resized_img(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);

    // 1. 包装源图像 buffer (假设 OpenCV 读取的是 BGR)
    // 必须确保 inputImg 内存是连续的，OpenCV 默认 Mat 只要不是裁剪的一般都是连续的
    rga_buffer_t src_rga = wrapbuffer_virtualaddr(
        (void*)inputImg.data, 
        inputImg.cols, 
        inputImg.rows, 
        RK_FORMAT_BGR_888 // 对应 CV_8UC3 (BGR)
    );

    // 2. 包装目标图像 buffer (RGB)
    rga_buffer_t dst_rga = wrapbuffer_virtualaddr(
        (void*)resized_img.data, 
        MODEL_WIDTH, 
        MODEL_HEIGHT, 
        RK_FORMAT_RGB_888 // 目标需要 RGB
    );

    // 3. 调用 RGA 执行 Resize + Format Conversion
    // imresize 内部会自动处理颜色空间转换
    IM_STATUS status = imresize(src_rga, dst_rga);
    if (status != IM_STATUS_SUCCESS) {
        std::cerr << "[RetinaFace] RGA resize failed: " << status << std::endl;
        return -1;
    }

    /* 用openCV：
    cv::Mat resized_img;
    cv::resize(inputImg, resized_img, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    */

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = MODEL_WIDTH * MODEL_HEIGHT * 3;
    inputs[0].buf = resized_img.data;
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_run(ctx, NULL);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 1; 
    }
    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    // ==========================================
    // 关键修正：解析官方格式 (Loc, Conf, Landm)
    // ==========================================
    // 官方模型通常顺序: 
    // outputs[0] -> Location [1, 4200, 4]
    // outputs[1] -> Score    [1, 4200, 2]
    // outputs[2] -> Landmark [1, 4200, 10]
    
    float* out_loc   = (float*)outputs[0].buf;
    float* out_score = (float*)outputs[1].buf;
    float* out_landm = (float*)outputs[2].buf;

    float variance[2] = {0.1f, 0.2f};
    float scale_w = (float)inputImg.cols; // 此时是相对坐标，直接乘原图宽高
    float scale_h = (float)inputImg.rows;

    std::vector<FaceInfo> proposals;
    int num_priors = priors.size(); // 4200

    // 遍历所有 Anchor
    for (int i = 0; i < num_priors; i++) {
        // Score: Index 1 is face confidence (Softmax后的结果)
        // 格式: [bg_score, face_score]
        float score = out_score[i * 2 + 1]; 
        
        if (score < 0.5f) continue; // CONF_THRESHOLD

        // Decode Box
        float* loc_ptr = out_loc + i * 4;
        float prior_cx = priors[i][0];
        float prior_cy = priors[i][1];
        float prior_w  = priors[i][2];
        float prior_h  = priors[i][3];

        float cx = prior_cx + loc_ptr[0] * variance[0] * prior_w;
        float cy = prior_cy + loc_ptr[1] * variance[0] * prior_h;
        float w  = prior_w  * exp(loc_ptr[2] * variance[1]);
        float h  = prior_h  * exp(loc_ptr[3] * variance[1]);

        // 转回原图坐标
        FaceInfo face;
        face.score = score;
        face.box.x = (cx - w / 2.0f) * scale_w;
        face.box.y = (cy - h / 2.0f) * scale_h;
        face.box.width = w * scale_w;
        face.box.height = h * scale_h;

        // Decode Landmarks
        float* landm_ptr = out_landm + i * 10;
        for (int k = 0; k < 5; k++) {
            float lcx = prior_cx + landm_ptr[k * 2] * variance[0] * prior_w;
            float lcy = prior_cy + landm_ptr[k * 2 + 1] * variance[0] * prior_h;
            face.landmarks[k].x = lcx * scale_w;
            face.landmarks[k].y = lcy * scale_h;
        }

        proposals.push_back(face);
    }

    // NMS (OpenCV 辅助)
    std::vector<int> indices;
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    for (const auto& p : proposals) {
        bboxes.push_back(p.box);
        scores.push_back(p.score);
    }
    
    cv::dnn::NMSBoxes(bboxes, scores, 0.5f, 0.4f, indices);

    for (int idx : indices) {
        faces.push_back(proposals[idx]);
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs);
    return 0;
}

cv::Mat RetinaFace::getAlignedFaceFromCamera(CameraManager& camera) {
    cv::Mat frame;
    if (!camera.getLatestFrame(frame) || frame.empty()) return cv::Mat();

    std::vector<FaceInfo> faces;
    detect(frame, faces);

    if (faces.empty()) return cv::Mat();

    // 取最大的脸
    auto bestFace = *std::max_element(faces.begin(), faces.end(), 
        [](const FaceInfo& a, const FaceInfo& b){ return a.box.area() < b.box.area(); });

    return preprocessFace(frame, bestFace.landmarks);
}

cv::Mat RetinaFace::preprocessFace(const cv::Mat& img, const cv::Point2f landmarks[5]) {
    cv::Mat aligned;
    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    for(int i=0; i<5; i++) src_pts.push_back(landmarks[i]);
    for(int i=0; i<5; i++) dst_pts.push_back(cv::Point2f(REFERENCE_PTS_112[i][0], REFERENCE_PTS_112[i][1]));

    cv::Mat M = cv::estimateAffinePartial2D(src_pts, dst_pts);
    if(!M.empty()) {
        cv::warpAffine(img, aligned, M, cv::Size(112, 112));
    }
    return aligned;
}