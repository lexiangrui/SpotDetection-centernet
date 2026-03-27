#include "spot_detector.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#include "opencv2/imgproc.hpp"

constexpr float SpotDetector::MEAN[3];
constexpr float SpotDetector::STD[3];

static std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

SpotDetector::SpotDetector() = default;

SpotDetector::~SpotDetector() { release(); }

int SpotDetector::init(const std::string& model_path, int input_w, int input_h) {
    input_w_ = input_w;
    input_h_ = input_h;

    auto model_data = load_file(model_path);
    if (model_data.empty()) {
        fprintf(stderr, "[ERR] Failed to load model: %s\n", model_path.c_str());
        return -1;
    }

    int ret = rknn_init(&ctx_, model_data.data(), model_data.size(), 0, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[ERR] rknn_init failed: %d\n", ret);
        return ret;
    }

    initialized_ = true;
    printf("[INFO] RKNN model loaded: %s\n", model_path.c_str());

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        fprintf(stderr, "[WARN] rknn_query IN_OUT_NUM failed: %d\n", ret);
    } else {
        printf("[INFO] Model I/O: %u inputs, %u outputs\n", io_num.n_input, io_num.n_output);
        if (io_num.n_output != 2) {
            fprintf(stderr, "[ERR] Expected 2 outputs (heatmap, reg), got %u\n", io_num.n_output);
            rknn_destroy(ctx_);
            initialized_ = false;
            return -1;
        }
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            rknn_tensor_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
            if (ret == 0) {
                printf("[INFO] Output[%u]: name=%s, dims=[%u,%u,%u,%u], type=%d, fmt=%d\n",
                       i, attr.name, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
                       attr.type, attr.fmt);
            }
        }
    }

    return 0;
}

void SpotDetector::release() {
    if (initialized_) {
        rknn_destroy(ctx_);
        initialized_ = false;
    }
}

cv::Mat SpotDetector::preprocess(const cv::Mat& image_bgr, ResizePadInfo& info) {
    info.orig_w = image_bgr.cols;
    info.orig_h = image_bgr.rows;
    info.dst_w = input_w_;
    info.dst_h = input_h_;

    float scale = std::min(static_cast<float>(info.dst_w) / info.orig_w,
                           static_cast<float>(info.dst_h) / info.orig_h);
    info.resized_w = std::max(static_cast<int>(std::round(info.orig_w * scale)), 1);
    info.resized_h = std::max(static_cast<int>(std::round(info.orig_h * scale)), 1);
    info.scale_x = static_cast<float>(info.resized_w) / info.orig_w;
    info.scale_y = static_cast<float>(info.resized_h) / info.orig_h;
    info.pad_left = std::max((info.dst_w - info.resized_w) / 2, 0);
    info.pad_top  = std::max((info.dst_h - info.resized_h) / 2, 0);

    cv::Mat rgb;
    cv::cvtColor(image_bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(info.resized_w, info.resized_h), 0, 0, cv::INTER_LINEAR);

    // NHWC float32, normalized with ImageNet mean/std
    // Padding must match training: normalize(0) = (0/255 - mean) / std = -mean/std
    cv::Mat canvas(info.dst_h, info.dst_w, CV_32FC3,
                   cv::Scalar(-MEAN[0] / STD[0], -MEAN[1] / STD[1], -MEAN[2] / STD[2]));

    for (int y = 0; y < info.resized_h; ++y) {
        const uint8_t* src_row = resized.ptr<uint8_t>(y);
        float* dst_row = canvas.ptr<float>(y + info.pad_top) + info.pad_left * 3;
        for (int x = 0; x < info.resized_w; ++x) {
            dst_row[x * 3 + 0] = (src_row[x * 3 + 0] / 255.0f - MEAN[0]) / STD[0];
            dst_row[x * 3 + 1] = (src_row[x * 3 + 1] / 255.0f - MEAN[1]) / STD[1];
            dst_row[x * 3 + 2] = (src_row[x * 3 + 2] / 255.0f - MEAN[2]) / STD[2];
        }
    }

    return canvas;
}

void SpotDetector::postprocess(const float* heatmap_data, const float* reg,
                               int out_h, int out_w,
                               const ResizePadInfo& info,
                               float score_threshold, int topk, int nms_kernel,
                               std::vector<Detection>& detections) {
    const int total = out_h * out_w;

    // ONNX/RKNN model already outputs post-sigmoid heatmap, no need to apply again
    const float* heatmap = heatmap_data;

    // Local NMS (max pooling)
    int pad = nms_kernel / 2;
    std::vector<float> suppressed(total, 0.0f);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            float val = heatmap[y * out_w + x];
            float max_val = val;
            for (int dy = -pad; dy <= pad; ++dy) {
                for (int dx = -pad; dx <= pad; ++dx) {
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < out_h && nx >= 0 && nx < out_w) {
                        max_val = std::max(max_val, heatmap[ny * out_w + nx]);
                    }
                }
            }
            suppressed[y * out_w + x] = (max_val == val) ? val : 0.0f;
        }
    }

    // Top-K selection
    std::vector<int> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    int k = std::min(topk, total);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&suppressed](int a, int b) { return suppressed[a] > suppressed[b]; });

    // Output scale for coordinate mapping (input -> output feature map)
    float out_scale_x = static_cast<float>(out_w) / info.dst_w;
    float out_scale_y = static_cast<float>(out_h) / info.dst_h;
    float out_pad_left = info.pad_left * out_scale_x;
    float out_pad_top  = info.pad_top  * out_scale_y;
    float out_sx = info.scale_x * out_scale_x;
    float out_sy = info.scale_y * out_scale_y;

    detections.clear();
    for (int i = 0; i < k; ++i) {
        int idx = indices[i];
        float score = suppressed[idx];
        if (score < score_threshold) break;

        int grid_x = idx % out_w;
        int grid_y = idx / out_w;

        float reg_x = reg[0 * total + idx];
        float reg_y = reg[1 * total + idx];

        float feat_x = grid_x + reg_x;
        float feat_y = grid_y + reg_y;

        // Map from output feature map back to original image coordinates
        float orig_x = (feat_x - out_pad_left) / std::max(out_sx, 1e-8f);
        float orig_y = (feat_y - out_pad_top)  / std::max(out_sy, 1e-8f);

        detections.push_back({orig_x, orig_y, score});
    }
}

std::vector<Detection> SpotDetector::detect(const cv::Mat& image_bgr,
                                            float score_threshold,
                                            int topk,
                                            int nms_kernel) {
    std::vector<Detection> detections;
    if (!initialized_) return detections;

    ResizePadInfo info{};
    cv::Mat input = preprocess(image_bgr, info);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = input_w_ * input_h_ * 3 * sizeof(float);
    inputs[0].buf = input.data;

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0) {
        fprintf(stderr, "[ERR] rknn_inputs_set failed: %d\n", ret);
        return detections;
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[ERR] rknn_run failed: %d\n", ret);
        return detections;
    }

    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    outputs[1].want_float = 1;

    ret = rknn_outputs_get(ctx_, 2, outputs, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[ERR] rknn_outputs_get failed: %d\n", ret);
        return detections;
    }

    int out_h = input_h_ / 4;
    int out_w = input_w_ / 4;
    const float* heatmap = static_cast<float*>(outputs[0].buf);
    const float* reg     = static_cast<float*>(outputs[1].buf);

    postprocess(heatmap, reg, out_h, out_w, info,
                score_threshold, topk, nms_kernel, detections);

    rknn_outputs_release(ctx_, 2, outputs);
    return detections;
}
