#include "spot_detector.h"

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_running(true);
std::atomic<bool> g_inference_enabled(true);

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) return;
        if (queue_.size() >= max_size_) {
            queue_.pop();
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_ = 0;
    bool closed_ = false;
};

struct PipelineData {
    GstElement* pipeline = nullptr;
    GstElement* appsrc   = nullptr;
    guint width  = 1280;
    guint height = 720;
    guint fps    = 30;
};

struct StreamConfig {
    std::string model_path = "./model/spot_centernet.rknn";
    std::string ip = "192.168.99.230";
    int camera_index = 22;
    float score_threshold = 0.1f;
    int topk = 256;
    int nms_kernel = 9;
    int frame_w = 1280;
    int frame_h = 720;
    int fps = 30;
    std::string video_mode = "30fps";
};

bool apply_video_mode(const std::string& mode, int& width, int& height, int& fps, std::string& resolved_name) {
    if (mode == "30fps" || mode == "720p30" || mode == "1280x720@30") {
        width = 1280;
        height = 720;
        fps = 30;
        resolved_name = "30fps";
        return true;
    }
    if (mode == "15fps" || mode == "2112x1568@15") {
        width = 2112;
        height = 1568;
        fps = 15;
        resolved_name = "15fps";
        return true;
    }
    return false;
}

void print_help(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [options]\n\n"
        << "Options:\n"
        << "  --model <path>         RKNN model path (default: ./model/spot_centernet.rknn)\n"
        << "  --ip <addr>            UDP target IP, port is fixed to 5000 (default: 192.168.99.230)\n"
        << "  --camera <index>       Camera index (default: 22)\n"
        << "  --threshold <float>    Detection score threshold (default: 0.1)\n"
        << "  --topk <int>           Top-K points kept before NMS (default: 256)\n"
        << "  --nms-kernel <int>     NMS kernel size, must be positive odd number (default: 9)\n"
        << "  --video-mode <mode>    Preset capture mode: 30fps | 15fps (default: 30fps)\n"
        << "  --width <int>          Custom capture width, use together with --height and --fps\n"
        << "  --height <int>         Custom capture height, use together with --width and --fps\n"
        << "  --fps <int>            Custom capture fps, use together with --width and --height\n"
        << "  --help, -h             Show this help message\n\n"
        << "Video modes:\n"
        << "  30fps                  1280x720 @ 30 fps\n"
        << "  15fps                  2112x1568 @ 15 fps\n\n"
        << "Examples:\n"
        << "  " << program << "\n"
        << "  " << program << " --model ./model/spot_centernet.rknn --ip 192.168.99.230\n"
        << "  " << program << " --camera 22 --threshold 0.15 --video-mode 15fps\n"
        << "  " << program << " --width 2112 --height 1568 --fps 15\n\n"
        << "Runtime:\n"
        << "  Input q + Enter to toggle inference on/off while keeping the stream alive.\n";
}

bool parse_int_arg(const std::string& text, const char* option_name, int& value) {
    try {
        size_t pos = 0;
        int parsed = std::stoi(text, &pos);
        if (pos != text.size()) {
            std::cerr << "[Args] Invalid integer for " << option_name << ": " << text << std::endl;
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        std::cerr << "[Args] Invalid integer for " << option_name << ": " << text << std::endl;
        return false;
    }
}

bool parse_float_arg(const std::string& text, const char* option_name, float& value) {
    try {
        size_t pos = 0;
        float parsed = std::stof(text, &pos);
        if (pos != text.size()) {
            std::cerr << "[Args] Invalid float for " << option_name << ": " << text << std::endl;
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        std::cerr << "[Args] Invalid float for " << option_name << ": " << text << std::endl;
        return false;
    }
}

bool require_value(int argc, char* argv[], int& index, const std::string& option, std::string& value) {
    if (index + 1 >= argc) {
        std::cerr << "[Args] Missing value for " << option << std::endl;
        return false;
    }
    const std::string next = argv[index + 1];
    if (next == "-h" || next.rfind("--", 0) == 0) {
        std::cerr << "[Args] Missing value for " << option << std::endl;
        return false;
    }
    value = next;
    ++index;
    return true;
}

enum class ParseResult {
    kOk,
    kHelp,
    kError,
};

ParseResult parse_args(int argc, char* argv[], StreamConfig& config) {
    bool width_set = false;
    bool height_set = false;
    bool fps_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;

        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return ParseResult::kHelp;
        }
        if (arg == "--model") {
            if (!require_value(argc, argv, i, arg, value)) return ParseResult::kError;
            config.model_path = value;
            continue;
        }
        if (arg == "--ip") {
            if (!require_value(argc, argv, i, arg, value)) return ParseResult::kError;
            config.ip = value;
            continue;
        }
        if (arg == "--camera") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--camera", config.camera_index)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (arg == "--threshold") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_float_arg(value, "--threshold", config.score_threshold)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (arg == "--topk") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--topk", config.topk)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (arg == "--nms-kernel") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--nms-kernel", config.nms_kernel)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (arg == "--video-mode") {
            if (!require_value(argc, argv, i, arg, value)) return ParseResult::kError;
            config.video_mode = value;
            continue;
        }
        if (arg == "--width") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--width", config.frame_w)) {
                return ParseResult::kError;
            }
            width_set = true;
            continue;
        }
        if (arg == "--height") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--height", config.frame_h)) {
                return ParseResult::kError;
            }
            height_set = true;
            continue;
        }
        if (arg == "--fps") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--fps", config.fps)) {
                return ParseResult::kError;
            }
            fps_set = true;
            continue;
        }

        std::cerr << "[Args] Unknown argument: " << arg << std::endl;
        print_help(argv[0]);
        return ParseResult::kError;
    }

    if (width_set || height_set || fps_set) {
        if (!(width_set && height_set && fps_set)) {
            std::cerr << "[Args] Custom video settings require --width, --height and --fps together" << std::endl;
            return ParseResult::kError;
        }
        config.video_mode = "custom";
    } else if (!apply_video_mode(config.video_mode, config.frame_w, config.frame_h, config.fps, config.video_mode)) {
        std::cerr << "[Args] Unsupported --video-mode: " << config.video_mode << std::endl;
        return ParseResult::kError;
    }

    if (config.camera_index < 0) {
        std::cerr << "[Args] --camera must be >= 0" << std::endl;
        return ParseResult::kError;
    }
    if (config.score_threshold < 0.0f) {
        std::cerr << "[Args] --threshold must be >= 0" << std::endl;
        return ParseResult::kError;
    }
    if (config.topk <= 0) {
        std::cerr << "[Args] --topk must be > 0" << std::endl;
        return ParseResult::kError;
    }
    if (config.nms_kernel <= 0 || (config.nms_kernel % 2) == 0) {
        std::cerr << "[Args] --nms-kernel must be a positive odd number" << std::endl;
        return ParseResult::kError;
    }
    if (config.frame_w <= 0 || config.frame_h <= 0 || config.fps <= 0) {
        std::cerr << "[Args] Video dimensions and fps must be > 0" << std::endl;
        return ParseResult::kError;
    }

    return ParseResult::kOk;
}

bool init_pipeline(PipelineData& pd, const std::string& ip) {
    std::string pipeline_str =
        "appsrc name=mysource ! "
        "video/x-raw,format=BGR,width="  + std::to_string(pd.width)  +
        ",height=" + std::to_string(pd.height) +
        ",framerate=" + std::to_string(pd.fps) + "/1 ! "
        "videoconvert ! "
        "mpph264enc ! "
        "rtph264pay config-interval=1 pt=96 ! "
        "udpsink host=" + ip + " port=5000";

    GError* error = nullptr;
    pd.pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pd.pipeline) {
        std::cerr << "[Pipeline] Failed to create: "
                  << (error ? error->message : "unknown error") << std::endl;
        g_clear_error(&error);
        return false;
    }

    pd.appsrc = gst_bin_get_by_name(GST_BIN(pd.pipeline), "mysource");
    if (!pd.appsrc) {
        std::cerr << "[Pipeline] appsrc not found" << std::endl;
        return false;
    }

    g_object_set(pd.appsrc,
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "format",      GST_FORMAT_TIME,
                 nullptr);

    gst_element_set_state(pd.pipeline, GST_STATE_PLAYING);
    return true;
}

void destroy_pipeline(PipelineData& pd) {
    if (pd.appsrc) {
        gst_app_src_end_of_stream(GST_APP_SRC(pd.appsrc));
        gst_object_unref(pd.appsrc);
        pd.appsrc = nullptr;
    }
    if (pd.pipeline) {
        gst_element_set_state(pd.pipeline, GST_STATE_NULL);
        gst_object_unref(pd.pipeline);
        pd.pipeline = nullptr;
    }
}

void on_signal(int) {
    g_running = false;
}

void control_thread() {
    std::string line;
    while (g_running.load() && std::getline(std::cin, line)) {
        if (line == "q" || line == "Q") {
            bool enabled = !g_inference_enabled.load();
            g_inference_enabled = enabled;
            std::cout << "[Control] Inference " << (enabled ? "enabled" : "disabled")
                      << " (stream keeps running)" << std::endl;
        }
    }
}

void draw_crosshair(cv::Mat& canvas, int cx, int cy,
                    int size, const cv::Scalar& color, int thickness) {
    cv::line(canvas, cv::Point(cx - size, cy), cv::Point(cx + size, cy),
             color, thickness, cv::LINE_AA);
    cv::line(canvas, cv::Point(cx, cy - size), cv::Point(cx, cy + size),
             color, thickness, cv::LINE_AA);
}

std::string make_detection_label(const cv::Mat& image, const Detection& det) {
    int x = std::clamp(static_cast<int>(std::lround(det.x)), 0, image.cols);
    int y = std::clamp(static_cast<int>(std::lround(static_cast<double>(image.rows) - det.y)), 0, image.rows);
    char label[96];
    snprintf(label, sizeof(label), "x=%d y=%d s=%.2f", x, y, det.score);
    return std::string(label);
}

void draw_label(cv::Mat& image, const std::string& label, cv::Point origin,
                double font_scale, int thickness, const cv::Scalar& color) {
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    int x = std::max(0, std::min(origin.x, image.cols - text_size.width - 4));
    int y = std::max(text_size.height + 4, std::min(origin.y, image.rows - baseline - 4));
    cv::rectangle(image,
                  cv::Point(x - 2, y - text_size.height - 2),
                  cv::Point(x + text_size.width + 2, y + baseline + 2),
                  cv::Scalar(0, 0, 0),
                  cv::FILLED);
    cv::putText(image, label, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv::LINE_AA);
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& dets) {
    cv::Scalar color(0, 255, 0);
    for (const auto& det : dets) {
        int cx = static_cast<int>(std::round(det.x));
        int cy = static_cast<int>(std::round(det.y));
        draw_crosshair(image, cx, cy, 6, color, 1);
        draw_label(image, make_detection_label(image, det), cv::Point(cx + 8, cy - 4), 0.4, 1, color);
    }
}

void capture_thread(BoundedQueue<cv::Mat>& raw_queue, int camera_index,
                    int frame_w, int frame_h, int fps) {
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "[Capture] Failed to open camera index " << camera_index << std::endl;
        g_running = false;
        raw_queue.close();
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_h);
    cap.set(cv::CAP_PROP_FPS, fps);

    std::cout << "[Capture] Started, camera=" << camera_index
              << " " << frame_w << "x" << frame_h << "@" << fps << std::endl;

    while (g_running.load()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[Capture] Empty frame, stopping" << std::endl;
            g_running = false;
            break;
        }
        raw_queue.push(std::move(frame));
    }

    raw_queue.close();
    std::cout << "[Capture] Stopped" << std::endl;
}

void detect_thread(BoundedQueue<cv::Mat>& raw_queue,
                   BoundedQueue<cv::Mat>& result_queue,
                   const std::string& model_path,
                   float score_threshold, int topk, int nms_kernel) {
    SpotDetector detector;
    int ret = detector.init(model_path);
    if (ret != 0) {
        std::cerr << "[Detect] Failed to init detector" << std::endl;
        g_running = false;
        result_queue.close();
        return;
    }
    std::cout << "[Detect] Detector ready: " << model_path << std::endl;

    size_t frame_count = 0;
    cv::Mat frame;
    while (g_running.load() && raw_queue.pop(frame)) {
        std::vector<Detection> dets;
        bool inference_enabled = g_inference_enabled.load();
        if (inference_enabled) {
            dets = detector.detect(frame, score_threshold, topk, nms_kernel);
            draw_detections(frame, dets);
        }

        if (++frame_count % 30 == 0) {
            std::cout << "[Buffer] raw=" << raw_queue.size()
                      << "/3  result=" << result_queue.size()
                      << "/3  spots=" << dets.size()
                      << "  infer=" << (inference_enabled ? "on" : "off")
                      << std::endl;
        }

        result_queue.push(std::move(frame));
    }

    result_queue.close();
    std::cout << "[Detect] Stopped" << std::endl;
}

void stream_thread(BoundedQueue<cv::Mat>& result_queue, const std::string& ip,
                   int frame_w, int frame_h, int fps) {
    gst_init(nullptr, nullptr);

    PipelineData pd;
    pd.width  = static_cast<guint>(frame_w);
    pd.height = static_cast<guint>(frame_h);
    pd.fps    = static_cast<guint>(fps);

    if (!init_pipeline(pd, ip)) {
        g_running = false;
        return;
    }

    std::cout << "[Stream] Streaming to " << ip << ":5000" << std::endl;

    guint64 frame_num = 0;
    cv::Mat frame;
    while (result_queue.pop(frame)) {
        if (frame.cols != frame_w || frame.rows != frame_h) {
            cv::resize(frame, frame, cv::Size(frame_w, frame_h));
        }

        const size_t data_size = frame.total() * frame.elemSize();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, data_size, nullptr);
        if (!buffer) {
            std::cerr << "[Stream] Buffer alloc failed" << std::endl;
            g_running = false;
            break;
        }

        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        std::memcpy(map.data, frame.data, data_size);
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = frame_num * gst_util_uint64_scale_int(1, GST_SECOND, pd.fps);
        GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, pd.fps);

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(pd.appsrc), buffer);
        if (ret != GST_FLOW_OK) {
            std::cerr << "[Stream] Push failed, ret=" << ret << std::endl;
            g_running = false;
            break;
        }

        ++frame_num;
    }

    destroy_pipeline(pd);
    std::cout << "[Stream] Stopped" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    g_running = true;
    g_inference_enabled = true;

    StreamConfig config;
    switch (parse_args(argc, argv, config)) {
    case ParseResult::kOk:
        break;
    case ParseResult::kHelp:
        return 0;
    case ParseResult::kError:
        return 1;
    }


    std::cout << "========================================\n"
              << "Real-time Spot Detection Stream\n"
              << "Model      : " << config.model_path << "\n"
              << "Target     : " << config.ip << ":5000\n"
              << "Camera     : " << config.camera_index << "\n"
              << "Video Mode : " << config.video_mode << "\n"
              << "Resolution : " << config.frame_w << "x" << config.frame_h << "@" << config.fps << "\n"
              << "Threshold  : " << config.score_threshold << "\n"
              << "TopK       : " << config.topk << "\n"
              << "NMS Kernel : " << config.nms_kernel << "\n"
              << "Input q + Enter to toggle inference\n"
              << "Press Ctrl+C to exit\n"
              << "========================================\n" << std::endl;

    BoundedQueue<cv::Mat> raw_queue(3);
    BoundedQueue<cv::Mat> result_queue(3);

    std::thread t_capture(capture_thread,
                          std::ref(raw_queue), config.camera_index,
                          config.frame_w, config.frame_h, config.fps);
    std::thread t_detect(detect_thread,
                         std::ref(raw_queue),
                         std::ref(result_queue),
                         std::cref(config.model_path),
                         config.score_threshold, config.topk, config.nms_kernel);
    std::thread t_stream(stream_thread,
                         std::ref(result_queue), std::cref(config.ip),
                         config.frame_w, config.frame_h, config.fps);
    std::thread t_control(control_thread);

    t_capture.join();
    raw_queue.close();
    t_detect.join();
    result_queue.close();
    t_stream.join();
    if (t_control.joinable()) {
        t_control.detach();
    }

    std::cout << "Done" << std::endl;
    return 0;
}
