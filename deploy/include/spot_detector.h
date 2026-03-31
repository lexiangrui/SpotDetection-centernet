#pragma once

#include <string>
#include <vector>

#include "rknn_api.h"
#include "opencv2/core.hpp"

struct Detection {
    float x;
    float y;
    float score;
};

struct ResizePadInfo {
    int orig_w;
    int orig_h;
    int dst_w;
    int dst_h;
    int resized_w;
    int resized_h;
    float scale_x;
    float scale_y;
    int pad_left;
    int pad_top;
};

class SpotDetector {
public:
    SpotDetector();
    ~SpotDetector();

    int init(const std::string& model_path, int input_w = 0, int input_h = 0);
    void release();

    std::vector<Detection> detect(const cv::Mat& image_bgr,
                                  float score_threshold = 0.1f,
                                  int topk = 256,
                                  int nms_kernel = 9);

private:
    cv::Mat preprocess(const cv::Mat& image_bgr, ResizePadInfo& info);
    void postprocess(const float* heatmap, const float* reg,
                     int out_h, int out_w,
                     const ResizePadInfo& info,
                     float score_threshold, int topk, int nms_kernel,
                     std::vector<Detection>& detections);

    rknn_context ctx_ = 0;
    bool initialized_ = false;
    bool is_int8_model_ = true;   // true=int8量化模型(输入uint8), false=fp32模型(输入float32)
    int input_w_ = 640;
    int input_h_ = 640;
    // ImageNet normalization params for fp32 model
    float mean_[3] = {0.485f, 0.456f, 0.406f};
    float std_[3]  = {0.229f, 0.224f, 0.225f};
};
