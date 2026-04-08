#include "spot_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#include "opencv2/imgproc.hpp"

static bool resolve_tensor_hw(const rknn_tensor_attr& attr, int& width, int& height) {
    if (attr.fmt == RKNN_TENSOR_NHWC) {
        height = static_cast<int>(attr.dims[1]);
        width  = static_cast<int>(attr.dims[2]);
    } else {
        height = static_cast<int>(attr.dims[2]);
        width  = static_cast<int>(attr.dims[3]);
    }
    return width > 0 && height > 0;
}

static bool resolve_tensor_channels(const rknn_tensor_attr& attr, int& channels) {
    if (attr.fmt == RKNN_TENSOR_NHWC) {
        channels = static_cast<int>(attr.dims[3]);
    } else {
        channels = static_cast<int>(attr.dims[1]);
    }
    return channels > 0;
}

static std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}

// Normalize uint8 NHWC RGB canvas to float32 in-place for fp32 model input.
static void normalize_canvas_uint8_to_float(const uint8_t* u8_data, float* float_data,
                                             int pixel_count,
                                             const float mean[3], const float std[3]) {
    const float inv_std[3] = {1.0f / std[0], 1.0f / std[1], 1.0f / std[2]};
    for (int i = 0; i < pixel_count; ++i) {
        int base = i * 3;
        float_data[base + 0] = (static_cast<float>(u8_data[base + 0]) / 255.0f - mean[0]) * inv_std[0];
        float_data[base + 1] = (static_cast<float>(u8_data[base + 1]) / 255.0f - mean[1]) * inv_std[1];
        float_data[base + 2] = (static_cast<float>(u8_data[base + 2]) / 255.0f - mean[2]) * inv_std[2];
    }
}

SpotDetector::SpotDetector() = default;

SpotDetector::~SpotDetector() { release(); }

int SpotDetector::init(const std::string& model_path, int input_w, int input_h) {
    input_w_ = input_w;
    input_h_ = input_h;
    output_w_ = 0;
    output_h_ = 0;
    heatmap_output_index_ = 0;
    reg_output_index_ = 1;
    reg_output_is_nhwc_ = false;

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
        if (io_num.n_input == 1) {
            rknn_tensor_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.index = 0;
            ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
            if (ret == 0) {
                int model_input_w = 0;
                int model_input_h = 0;
                if (resolve_tensor_hw(attr, model_input_w, model_input_h)) {
                    input_w_ = model_input_w;
                    input_h_ = model_input_h;
                }

                // Prefer the runtime tensor dtype over qnt_type when deciding how to feed input.
                // Some "fp" RKNN exports still carry affine quantization metadata while exposing
                // a FLOAT16/FLOAT32 input tensor, and those models must still use float input.
                const bool input_is_float =
                    (attr.type == RKNN_TENSOR_FLOAT16 || attr.type == RKNN_TENSOR_FLOAT32);
                const bool input_is_integer =
                    (attr.type == RKNN_TENSOR_UINT8 || attr.type == RKNN_TENSOR_INT8);
                if (input_is_float) {
                    is_int8_model_ = false;
                } else if (input_is_integer) {
                    is_int8_model_ = true;
                } else {
                    is_int8_model_ = (attr.qnt_type != RKNN_TENSOR_QNT_NONE);
                }

                printf("[INFO] Input[0]: name=%s, dims=[%u,%u,%u,%u], type=%d, fmt=%d, qnt=%d, zp=%d, scale=%.4f\n",
                       attr.name, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
                       attr.type, attr.fmt, attr.qnt_type, attr.zp, attr.scale);
                printf("[INFO] Model type: %s  (qnt_type=%d)\n",
                       is_int8_model_ ? "integer input" : "floating-point input",
                       attr.qnt_type);
                printf("[INFO] Detector input size: %dx%d\n", input_w_, input_h_);
            } else if (input_w_ <= 0 || input_h_ <= 0) {
                fprintf(stderr, "[ERR] Failed to query input tensor and no fallback size provided\n");
                rknn_destroy(ctx_);
                initialized_ = false;
                return -1;
            }
        }
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
                int out_w = 0;
                int out_h = 0;
                int out_c = 0;
                resolve_tensor_hw(attr, out_w, out_h);
                resolve_tensor_channels(attr, out_c);

                if (out_c == 1) {
                    heatmap_output_index_ = static_cast<int>(i);
                    output_w_ = out_w;
                    output_h_ = out_h;
                } else if (out_c == 2) {
                    reg_output_index_ = static_cast<int>(i);
                    reg_output_is_nhwc_ = (attr.fmt == RKNN_TENSOR_NHWC);
                    if (output_w_ == 0 || output_h_ == 0) {
                        output_w_ = out_w;
                        output_h_ = out_h;
                    } else if (output_w_ != out_w || output_h_ != out_h) {
                        fprintf(stderr,
                                "[ERR] Output shape mismatch: heatmap=%dx%d, reg=%dx%d\n",
                                output_w_, output_h_, out_w, out_h);
                        rknn_destroy(ctx_);
                        initialized_ = false;
                        return -1;
                    }
                }
                printf("[INFO] Output[%u]: name=%s, dims=[%u,%u,%u,%u], type=%d, fmt=%d\n",
                       i, attr.name, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
                       attr.type, attr.fmt);
            }
        }
    }

    if (input_w_ <= 0 || input_h_ <= 0) {
        fprintf(stderr, "[ERR] Failed to resolve model input size\n");
        rknn_destroy(ctx_);
        initialized_ = false;
        return -1;
    }
    if (output_w_ <= 0 || output_h_ <= 0) {
        output_w_ = std::max(input_w_ / 4, 1);
        output_h_ = std::max(input_h_ / 4, 1);
        fprintf(stderr,
                "[WARN] Failed to resolve output tensor size, fallback to input/4 = %dx%d\n",
                output_w_, output_h_);
    }

    printf("[INFO] Detector output size: %dx%d (heatmap idx=%d, reg idx=%d, reg fmt=%s)\n",
           output_w_, output_h_,
           heatmap_output_index_, reg_output_index_,
           reg_output_is_nhwc_ ? "NHWC" : "NCHW");

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

    cv::Mat canvas(info.dst_h, info.dst_w, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect(info.pad_left, info.pad_top, info.resized_w, info.resized_h)));
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

        float reg_x = 0.0f;
        float reg_y = 0.0f;
        if (reg_output_is_nhwc_) {
            reg_x = reg[idx * 2 + 0];
            reg_y = reg[idx * 2 + 1];
        } else {
            reg_x = reg[0 * total + idx];
            reg_y = reg[1 * total + idx];
        }

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
    cv::Mat canvas = preprocess(image_bgr, info);  // canvas: uint8 RGB [H, W, 3], NHWC layout

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].pass_through = 0;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    if (is_int8_model_) {
        // INT8 quantized model: input uint8, mean/std normalization fused into NPU graph.
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = static_cast<uint32_t>(canvas.total() * canvas.elemSize());
        inputs[0].buf  = canvas.data;
    } else {
        // FP32 non-quantized model: input float32, normalization done on CPU.
        // Allocate float buffer (reused across calls, minimal allocation cost vs inference cost).
        static thread_local std::vector<float> float_buf;
        float_buf.resize(canvas.total() * 3);
        normalize_canvas_uint8_to_float(canvas.data, float_buf.data(),
                                        static_cast<int>(canvas.total()),
                                        mean_, std_);
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].size = static_cast<uint32_t>(float_buf.size() * sizeof(float));
        inputs[0].buf  = float_buf.data();
    }

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

    int out_h = output_h_;
    int out_w = output_w_;
    const float* heatmap = static_cast<float*>(outputs[heatmap_output_index_].buf);
    const float* reg     = static_cast<float*>(outputs[reg_output_index_].buf);

    postprocess(heatmap, reg, out_h, out_w, info,
                score_threshold, topk, nms_kernel, detections);

    rknn_outputs_release(ctx_, 2, outputs);
    return detections;
}
