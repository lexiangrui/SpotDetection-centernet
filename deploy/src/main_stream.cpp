#include "spot_detector.h"

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_running(true);

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

void draw_crosshair(cv::Mat& canvas, int cx, int cy,
                    int size, const cv::Scalar& color, int thickness) {
    cv::line(canvas, cv::Point(cx - size, cy), cv::Point(cx + size, cy),
             color, thickness, cv::LINE_AA);
    cv::line(canvas, cv::Point(cx, cy - size), cv::Point(cx, cy + size),
             color, thickness, cv::LINE_AA);
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& dets) {
    cv::Scalar color(0, 255, 0);
    for (const auto& det : dets) {
        int cx = static_cast<int>(std::round(det.x));
        int cy = static_cast<int>(std::round(det.y));
        draw_crosshair(image, cx, cy, 6, color, 1);
        char label[32];
        snprintf(label, sizeof(label), "%.2f", det.score);
        cv::putText(image, label,
                    cv::Point(cx + 8, cy - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
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
        auto dets = detector.detect(frame, score_threshold, topk, nms_kernel);
        draw_detections(frame, dets);

        if (++frame_count % 30 == 0) {
            std::cout << "[Buffer] raw=" << raw_queue.size()
                      << "/3  result=" << result_queue.size()
                      << "/3  spots=" << dets.size()
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

    std::string model_path = "./model/spot_centernet.rknn";
    std::string ip = "192.168.99.230";
    int camera_index = 22;
    float score_threshold = 0.6f;
    int topk = 256;
    int nms_kernel = 5;
    int frame_w = 1280;
    int frame_h = 720;
    int fps = 30;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) ip = argv[2];
    if (argc >= 4) camera_index = std::stoi(argv[3]);
    if (argc >= 5) score_threshold = std::stof(argv[4]);
    if (argc >= 6) topk = std::stoi(argv[5]);
    if (argc >= 7) nms_kernel = std::stoi(argv[6]);

    std::cout << "========================================\n"
              << "Real-time Spot Detection Stream\n"
              << "Model      : " << model_path << "\n"
              << "Target     : " << ip << ":5000\n"
              << "Camera     : " << camera_index << "\n"
              << "Resolution : " << frame_w << "x" << frame_h << "@" << fps << "\n"
              << "Threshold  : " << score_threshold << "\n"
              << "TopK       : " << topk << "\n"
              << "NMS Kernel : " << nms_kernel << "\n"
              << "Press Ctrl+C to exit\n"
              << "========================================\n" << std::endl;

    BoundedQueue<cv::Mat> raw_queue(3);
    BoundedQueue<cv::Mat> result_queue(3);

    std::thread t_capture(capture_thread,
                          std::ref(raw_queue), camera_index,
                          frame_w, frame_h, fps);
    std::thread t_detect(detect_thread,
                         std::ref(raw_queue),
                         std::ref(result_queue),
                         std::cref(model_path),
                         score_threshold, topk, nms_kernel);
    std::thread t_stream(stream_thread,
                         std::ref(result_queue), std::cref(ip),
                         frame_w, frame_h, fps);

    t_capture.join();
    raw_queue.close();
    t_detect.join();
    result_queue.close();
    t_stream.join();

    std::cout << "Done" << std::endl;
    return 0;
}
