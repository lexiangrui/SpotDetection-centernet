#include "spot_detector.h"

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <ctime>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <linux/media-bus-format.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace {

std::atomic<bool> g_running(true);
std::atomic<bool> g_inference_enabled(true);
std::atomic<bool> g_grid_enabled(true);
std::mutex g_snapshot_mutex;

struct DetectionSnapshot {
    uint64_t frame_index = 0;
    int frame_w = 0;
    int frame_h = 0;
    bool inference_enabled = false;
    std::vector<Detection> detections;
};

DetectionSnapshot g_latest_snapshot;

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
    std::string model_path = "./model/spot_centernet_resnet18_int8.rknn";
    std::string ip = "192.168.99.230";
    int camera_index = 22;
    float score_threshold = 0.3f;
    int topk = 256;
    int nms_kernel = 5;
    int grid_step = 100;
    int frame_w = 1920;
    int frame_h = 1080;
    int fps = 30;
    std::string resolution_preset = "low";
};

struct V4L2MappedBuffer {
    void* data = nullptr;
    size_t length = 0;
};

int retry_ioctl(int fd, unsigned long request, void* arg) {
    int ret = 0;
    do {
        ret = ioctl(fd, request, arg);
    } while (ret < 0 && errno == EINTR);
    return ret;
}

std::string camera_device_path(int camera_index) {
    return "/dev/video" + std::to_string(camera_index);
}

std::string fourcc_to_string(uint32_t fourcc) {
    char text[5] = {
        static_cast<char>(fourcc & 0xff),
        static_cast<char>((fourcc >> 8) & 0xff),
        static_cast<char>((fourcc >> 16) & 0xff),
        static_cast<char>((fourcc >> 24) & 0xff),
        '\0',
    };
    return std::string(text);
}

bool needs_high_res_pipeline(int width, int height) {
    return width > 2112 || height > 1568;
}

bool set_subdev_format(const std::string& device_path, uint32_t pad,
                       uint32_t code, int width, int height,
                       const char* label) {
    int fd = open(device_path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::cerr << "[Subdev] Failed to open " << device_path
                  << " for " << label << ": " << std::strerror(errno) << std::endl;
        return false;
    }

    v4l2_subdev_format fmt{};
    fmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    fmt.pad = pad;
    fmt.format.width = static_cast<uint32_t>(width);
    fmt.format.height = static_cast<uint32_t>(height);
    fmt.format.code = code;
    fmt.format.field = V4L2_FIELD_NONE;

    if (retry_ioctl(fd, VIDIOC_SUBDEV_S_FMT, &fmt) < 0) {
        std::cerr << "[Subdev] VIDIOC_SUBDEV_S_FMT failed for " << label
                  << " on " << device_path << " pad " << pad
                  << ": " << std::strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    close(fd);
    std::cout << "[Subdev] " << label << " -> " << device_path
              << " pad=" << pad
              << " code=0x" << std::hex << fmt.format.code << std::dec
              << " size=" << fmt.format.width << "x" << fmt.format.height
              << std::endl;
    return true;
}

bool configure_high_res_pipeline(const StreamConfig& config) {
    if (!needs_high_res_pipeline(config.frame_w, config.frame_h)) {
        return true;
    }

    std::cout << "[Subdev] Pre-configuring OV13850/RKISP pipeline for "
              << config.frame_w << "x" << config.frame_h << std::endl;

    // This deployment is fixed to the RK3576 board where:
    // OV13850 -> csi2-dphy3 -> rockchip-mipi-csi2 -> rkcif-mipi-lvds3 -> rkisp-vir1
    static const std::string kSensorSubdev = "/dev/v4l-subdev2";
    static const std::string kDphySubdev = "/dev/v4l-subdev1";
    static const std::string kCsiSubdev = "/dev/v4l-subdev0";
    static const std::string kCifSubdev = "/dev/v4l-subdev5";
    static const std::string kIspSubdev = "/dev/v4l-subdev4";

    const uint32_t raw_code = MEDIA_BUS_FMT_SBGGR10_1X10;
    const uint32_t isp_output_code = MEDIA_BUS_FMT_YUYV8_2X8;

    // On this BSP the source pads of DPHY/CSI reject explicit S_FMT with EINVAL.
    // The pipeline still negotiates correctly as long as the sensor, sink pads,
    // CIF source, and ISP pads are configured before opening /dev/video22.
    const bool ok =
        set_subdev_format(kSensorSubdev, 0, raw_code,
                          config.frame_w, config.frame_h, "sensor source") &&
        set_subdev_format(kDphySubdev, 0, raw_code,
                          config.frame_w, config.frame_h, "dphy sink") &&
        set_subdev_format(kCsiSubdev, 0, raw_code,
                          config.frame_w, config.frame_h, "csi sink") &&
        set_subdev_format(kCifSubdev, 0, raw_code,
                          config.frame_w, config.frame_h, "cif source") &&
        set_subdev_format(kIspSubdev, 0, raw_code,
                          config.frame_w, config.frame_h, "isp sink") &&
        set_subdev_format(kIspSubdev, 2, isp_output_code,
                          config.frame_w, config.frame_h, "isp mainpath source");

    if (!ok) {
        std::cerr << "[Subdev] Failed to switch pipeline to "
                  << config.frame_w << "x" << config.frame_h << std::endl;
        return false;
    }

    // Let the media pipeline settle before opening the video node.
    usleep(200 * 1000);
    return true;
}

class V4L2Capture {
public:
    ~V4L2Capture() {
        close_device();
    }

    bool open_device(int camera_index, int requested_w, int requested_h, int requested_fps) {
        close_device();

        requested_w_ = requested_w;
        requested_h_ = requested_h;
        requested_fps_ = requested_fps;
        device_path_ = camera_device_path(camera_index);

        fd_ = open(device_path_.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0) {
            std::cerr << "[Capture] Failed to open " << device_path_
                      << ": " << std::strerror(errno) << std::endl;
            return false;
        }

        if (!configure_format() || !configure_fps() || !init_mmap(4) || !start_streaming()) {
            close_device();
            return false;
        }
        return true;
    }

    void close_device() {
        if (fd_ < 0) return;

        if (streaming_) {
            v4l2_buf_type type = buffer_type_;
            if (retry_ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
                std::cerr << "[Capture] VIDIOC_STREAMOFF failed for " << device_path_
                          << ": " << std::strerror(errno) << std::endl;
            }
            streaming_ = false;
        }

        for (auto& buffer : buffers_) {
            if (buffer.data && buffer.length > 0) {
                munmap(buffer.data, buffer.length);
                buffer.data = nullptr;
                buffer.length = 0;
            }
        }
        buffers_.clear();

        v4l2_requestbuffers req{};
        req.count = 0;
        req.type = buffer_type_;
        req.memory = V4L2_MEMORY_MMAP;
        retry_ioctl(fd_, VIDIOC_REQBUFS, &req);

        close(fd_);
        fd_ = -1;
    }

    bool read_frame(cv::Mat& bgr_frame) {
        if (fd_ < 0) return false;

        v4l2_buffer buf{};
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        buf.type = buffer_type_;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.length = 1;
        buf.m.planes = planes;

        if (retry_ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            std::cerr << "[Capture] VIDIOC_DQBUF failed for " << device_path_
                      << ": " << std::strerror(errno) << std::endl;
            return false;
        }

        bool ok = false;
        if (buf.index < buffers_.size()) {
            ok = convert_buffer(buf.index, bgr_frame);
        } else {
            std::cerr << "[Capture] Invalid buffer index from " << device_path_
                      << ": " << buf.index << std::endl;
        }

        if (!queue_buffer(buf.index)) {
            return false;
        }
        return ok;
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int fps() const { return fps_; }
    const std::string& device_path() const { return device_path_; }
    std::string pixel_format_name() const { return fourcc_to_string(pixel_format_); }

private:
    bool configure_format() {
        v4l2_capability cap{};
        if (retry_ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            std::cerr << "[Capture] VIDIOC_QUERYCAP failed for " << device_path_
                      << ": " << std::strerror(errno) << std::endl;
            return false;
        }

        const uint32_t capture_caps = cap.device_caps != 0 ? cap.device_caps : cap.capabilities;
        if ((capture_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) == 0) {
            std::cerr << "[Capture] Device " << device_path_
                      << " does not support multi-plane capture" << std::endl;
            return false;
        }

        const bool prefer_nv12 = needs_high_res_pipeline(requested_w_, requested_h_);
        const uint32_t preferred_formats[] = {
            prefer_nv12 ? V4L2_PIX_FMT_NV12 : V4L2_PIX_FMT_UYVY,
            prefer_nv12 ? V4L2_PIX_FMT_UYVY : V4L2_PIX_FMT_YUYV,
            prefer_nv12 ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_NV12,
        };

        bool configured = false;
        int last_errno = 0;
        for (uint32_t requested_format : preferred_formats) {
            v4l2_format fmt{};
            fmt.type = buffer_type_;
            fmt.fmt.pix_mp.width = static_cast<uint32_t>(requested_w_);
            fmt.fmt.pix_mp.height = static_cast<uint32_t>(requested_h_);
            fmt.fmt.pix_mp.pixelformat = requested_format;
            fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
            fmt.fmt.pix_mp.num_planes = 1;

            if (retry_ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
                last_errno = errno;
                continue;
            }

            width_ = static_cast<int>(fmt.fmt.pix_mp.width);
            height_ = static_cast<int>(fmt.fmt.pix_mp.height);
            pixel_format_ = fmt.fmt.pix_mp.pixelformat;
            bytes_per_line_ = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
            if (bytes_per_line_ == 0) {
                bytes_per_line_ = (pixel_format_ == V4L2_PIX_FMT_NV12)
                    ? static_cast<uint32_t>(width_)
                    : static_cast<uint32_t>(width_ * 2);
            }

            if (pixel_format_ == V4L2_PIX_FMT_UYVY ||
                pixel_format_ == V4L2_PIX_FMT_YUYV ||
                pixel_format_ == V4L2_PIX_FMT_NV12) {
                configured = true;
                break;
            }

            std::cerr << "[Capture] Driver returned unsupported pixel format from "
                      << device_path_ << ": " << pixel_format_name() << std::endl;
        }

        if (!configured) {
            std::cerr << "[Capture] VIDIOC_S_FMT failed for " << device_path_
                      << " with all supported pixel formats"
                      << (last_errno ? std::string(": ") + std::strerror(last_errno) : std::string())
                      << std::endl;
            return false;
        }

        if (width_ != requested_w_ || height_ != requested_h_) {
            std::cerr << "[Capture] Requested " << requested_w_ << "x" << requested_h_
                      << " but driver configured " << width_ << "x" << height_
                      << " on " << device_path_ << std::endl;
        }
        return true;
    }

    bool configure_fps() {
        fps_ = requested_fps_;

        v4l2_streamparm parm{};
        parm.type = buffer_type_;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = std::max(requested_fps_, 1);

        if (retry_ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
            std::cerr << "[Capture] VIDIOC_S_PARM unsupported on " << device_path_
                      << ", keeping driver default fps: " << std::strerror(errno) << std::endl;
            return true;
        }

        if (parm.parm.capture.timeperframe.numerator > 0 &&
            parm.parm.capture.timeperframe.denominator > 0) {
            fps_ = static_cast<int>(std::lround(
                static_cast<double>(parm.parm.capture.timeperframe.denominator) /
                parm.parm.capture.timeperframe.numerator));
        }
        return true;
    }

    bool init_mmap(uint32_t buffer_count) {
        v4l2_requestbuffers req{};
        req.count = buffer_count;
        req.type = buffer_type_;
        req.memory = V4L2_MEMORY_MMAP;

        if (retry_ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            std::cerr << "[Capture] VIDIOC_REQBUFS failed for " << device_path_
                      << ": " << std::strerror(errno) << std::endl;
            return false;
        }
        if (req.count < 2) {
            std::cerr << "[Capture] Insufficient mmap buffers from " << device_path_
                      << ": " << req.count << std::endl;
            return false;
        }

        buffers_.assign(req.count, {});
        for (uint32_t i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            v4l2_plane planes[VIDEO_MAX_PLANES]{};
            buf.type = buffer_type_;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            buf.length = 1;
            buf.m.planes = planes;

            if (retry_ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                std::cerr << "[Capture] VIDIOC_QUERYBUF failed for " << device_path_
                          << ": " << std::strerror(errno) << std::endl;
                return false;
            }

            void* data = mmap(nullptr, buf.m.planes[0].length,
                              PROT_READ | PROT_WRITE, MAP_SHARED,
                              fd_, buf.m.planes[0].m.mem_offset);
            if (data == MAP_FAILED) {
                std::cerr << "[Capture] mmap failed for " << device_path_
                          << ": " << std::strerror(errno) << std::endl;
                return false;
            }

            buffers_[i].data = data;
            buffers_[i].length = buf.m.planes[0].length;
        }
        return true;
    }

    bool start_streaming() {
        for (uint32_t i = 0; i < buffers_.size(); ++i) {
            if (!queue_buffer(i)) return false;
        }

        v4l2_buf_type type = buffer_type_;
        if (retry_ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            std::cerr << "[Capture] VIDIOC_STREAMON failed for " << device_path_
                      << ": " << std::strerror(errno) << std::endl;
            return false;
        }

        streaming_ = true;
        return true;
    }

    bool queue_buffer(uint32_t index) {
        v4l2_buffer buf{};
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        buf.type = buffer_type_;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = index;
        buf.length = 1;
        buf.m.planes = planes;

        if (retry_ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "[Capture] VIDIOC_QBUF failed for " << device_path_
                      << ": " << std::strerror(errno) << std::endl;
            return false;
        }
        return true;
    }

    bool convert_buffer(uint32_t index, cv::Mat& bgr_frame) const {
        if (index >= buffers_.size()) return false;

        const auto& buffer = buffers_[index];
        if (pixel_format_ == V4L2_PIX_FMT_NV12) {
            cv::Mat yuv(height_ * 3 / 2, width_, CV_8UC1, buffer.data, bytes_per_line_);
            cv::cvtColor(yuv, bgr_frame, cv::COLOR_YUV2BGR_NV12);
            return true;
        }

        cv::Mat yuv(height_, width_, CV_8UC2, buffer.data, bytes_per_line_);
        if (pixel_format_ == V4L2_PIX_FMT_UYVY) {
            cv::cvtColor(yuv, bgr_frame, cv::COLOR_YUV2BGR_UYVY);
            return true;
        }
        if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
            cv::cvtColor(yuv, bgr_frame, cv::COLOR_YUV2BGR_YUY2);
            return true;
        }
        return false;
    }

    int fd_ = -1;
    std::string device_path_;
    int requested_w_ = 0;
    int requested_h_ = 0;
    int requested_fps_ = 0;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 0;
    uint32_t pixel_format_ = V4L2_PIX_FMT_UYVY;
    uint32_t bytes_per_line_ = 0;
    v4l2_buf_type buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    std::vector<V4L2MappedBuffer> buffers_;
    bool streaming_ = false;
};

bool apply_resolution_preset(const std::string& mode, int& width, int& height, std::string& resolved_name) {
    if (mode == "low" || mode == "1080p" || mode == "1920x1080") {
        width = 1920;
        height = 1080;
        resolved_name = "low";
        return true;
    }
    if (mode == "medium" || mode == "2112x1568") {
        width = 2112;
        height = 1568;
        resolved_name = "medium";
        return true;
    }
    if (mode == "high" || mode == "4224x3136") {
        width = 4224;
        height = 3136;
        resolved_name = "high";
        return true;
    }
    return false;
}

void print_help(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [options]\n\n"
        << "Options:\n"
        << "  --model <path>         RKNN model path (default: ./model/spot_centernet_resnet18_int8.rknn)\n"
        << "  --ip <addr>            UDP target IP, port is fixed to 5000 (default: 192.168.99.230)\n"
        << "  --camera <index>       Camera index (default: 22)\n"
        << "  --threshold <float>    Detection score threshold (default: 0.3)\n"
        << "  --topk <int>           Top-K points kept before NMS (default: 256)\n"
        << "  --nms-kernel <int>     NMS kernel size, must be positive odd number (default: 5)\n"
        << "  --grid-step <int>      Coordinate grid spacing in pixels, <= 0 disables (default: 100)\n"
        << "  --video-mode <mode>    Preset resolution: low | medium | high (default: low)\n"
        << "  --resolution <mode>    Alias of --video-mode\n"
        << "  --width <int>          Custom capture width, use together with --height\n"
        << "  --height <int>         Custom capture height, use together with --width\n"
        << "  --fps <int>            Capture fps (default: 30)\n"
        << "  --help, -h             Show this help message\n\n"
        << "Resolution presets:\n"
        << "  low                    1920x1080\n"
        << "  medium                 2112x1568\n"
        << "  high                   4224x3136\n\n"
        << "Examples:\n"
        << "  " << program << "\n"
        << "  " << program << " --model ./model/spot_centernet_resnet18_int8.rknn --ip 192.168.99.230\n"
        << "  " << program << " --camera 22 --threshold 0.3 --video-mode low --fps 30\n"
        << "  " << program << " --video-mode medium --fps 30 --grid-step 200\n"
        << "  " << program << " --video-mode high --fps 15 --grid-step 200\n"
        << "  " << program << " --width 2112 --height 1568 --fps 15\n"
        << "  " << program << " --width 4224 --height 3136 --fps 15\n\n"
        << "Runtime:\n"
        << "  Input q + Enter to toggle inference on/off while keeping the stream alive.\n"
        << "  Input w + Enter to toggle the coordinate grid on/off.\n"
        << "  Input e + Enter to export current frame spot coordinates.\n";
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
        if (arg == "--grid-step") {
            if (!require_value(argc, argv, i, arg, value) ||
                !parse_int_arg(value, "--grid-step", config.grid_step)) {
                return ParseResult::kError;
            }
            continue;
        }
        if (arg == "--video-mode" || arg == "--resolution") {
            if (!require_value(argc, argv, i, arg, value)) return ParseResult::kError;
            config.resolution_preset = value;
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
            continue;
        }

        std::cerr << "[Args] Unknown argument: " << arg << std::endl;
        print_help(argv[0]);
        return ParseResult::kError;
    }

    if (width_set || height_set) {
        if (!(width_set && height_set)) {
            std::cerr << "[Args] Custom resolution requires --width and --height together" << std::endl;
            return ParseResult::kError;
        }
        config.resolution_preset = "custom";
    } else if (!apply_resolution_preset(config.resolution_preset, config.frame_w, config.frame_h, config.resolution_preset)) {
        std::cerr << "[Args] Unsupported --video-mode: " << config.resolution_preset << std::endl;
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
    if (config.grid_step < 0) {
        std::cerr << "[Args] --grid-step must be >= 0" << std::endl;
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
                 "is-live",     TRUE,
                 "block",       TRUE,
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

bool save_detection_snapshot(const DetectionSnapshot& snapshot, std::string& output_path) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);

    const std::filesystem::path output_dir("outputs");
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        return false;
    }

    std::ostringstream name_builder;
    name_builder << "spot_coords_frame_" << snapshot.frame_index << "_"
                 << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << ".txt";
    output_path = (output_dir / name_builder.str()).string();

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        return false;
    }

    ofs << "frame_index=" << snapshot.frame_index << "\n";
    ofs << "frame_size=" << snapshot.frame_w << "x" << snapshot.frame_h << "\n";
    ofs << "inference_enabled=" << (snapshot.inference_enabled ? "true" : "false") << "\n";
    ofs << "spot_count=" << snapshot.detections.size() << "\n\n";

    for (const auto& det : snapshot.detections) {
        const int x = std::clamp(static_cast<int>(std::lround(det.x)), 0, snapshot.frame_w);
        const int y = std::clamp(
            static_cast<int>(std::lround(static_cast<double>(snapshot.frame_h) - det.y)),
            0,
            snapshot.frame_h
        );
        ofs << "spot_id=" << det.id
            << " x=" << x
            << " y=" << y
            << " score=" << std::fixed << std::setprecision(4) << det.score
            << "\n";
    }

    return true;
}

void control_thread() {
    std::string line;
    while (g_running.load() && std::getline(std::cin, line)) {
        if (line == "q" || line == "Q") {
            bool enabled = !g_inference_enabled.load();
            g_inference_enabled = enabled;
            std::cout << "[Control] Inference " << (enabled ? "enabled" : "disabled")
                      << " (stream keeps running)" << std::endl;
        } else if (line == "w" || line == "W") {
            bool enabled = !g_grid_enabled.load();
            g_grid_enabled = enabled;
            std::cout << "[Control] Coordinate grid " << (enabled ? "enabled" : "disabled")
                      << std::endl;
        } else if (line == "e" || line == "E") {
            DetectionSnapshot snapshot;
            {
                std::lock_guard<std::mutex> lock(g_snapshot_mutex);
                snapshot = g_latest_snapshot;
            }

            if (snapshot.frame_w <= 0 || snapshot.frame_h <= 0) {
                std::cout << "[Control] No processed frame available yet" << std::endl;
                continue;
            }

            std::string output_path;
            if (!save_detection_snapshot(snapshot, output_path)) {
                std::cerr << "[Control] Failed to write coordinate file" << std::endl;
                continue;
            }

            std::cout << "[Control] Exported " << snapshot.detections.size()
                      << " spots from frame " << snapshot.frame_index
                      << " to " << output_path << std::endl;
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

void draw_label(cv::Mat& image, const std::string& label, cv::Point origin,
                double font_scale, int thickness, const cv::Scalar& color,
                bool align_right = false) {
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    if (align_right) {
        origin.x -= text_size.width;
    }
    int x = std::max(0, std::min(origin.x, image.cols - text_size.width - 4));
    int y = std::max(text_size.height + 4, std::min(origin.y, image.rows - baseline - 4));
    cv::putText(image, label, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv::LINE_AA);
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& dets) {
    const int min_side = std::min(image.rows, image.cols);
    const int marker_size = std::max(5, static_cast<int>(std::round(min_side * 0.009)));
    const int marker_thickness = std::max(1, static_cast<int>(std::round(marker_size / 5.0)));
    const double font_scale = std::max(0.22, marker_size / 22.0);
    const int text_thickness = std::min(3, std::max(1, static_cast<int>(std::round(min_side / 1200.0))));
    cv::Scalar color(0, 255, 0);
    for (const auto& det : dets) {
        int cx = static_cast<int>(std::round(det.x));
        int cy = static_cast<int>(std::round(det.y));
        int display_x = std::clamp(static_cast<int>(std::lround(det.x)), 0, image.cols);
        int display_y = std::clamp(static_cast<int>(std::lround(static_cast<double>(image.rows) - det.y)), 0, image.rows);
        int offset = std::max(4, marker_size - 1);
        int lower_y = cy + std::max(8, marker_size + 2);

        draw_crosshair(image, cx, cy, marker_size, color, marker_thickness);
        draw_label(image, "#" + std::to_string(det.id),
                   cv::Point(cx - offset, cy - offset),
                   font_scale, text_thickness, color, true);
        draw_label(image, "(" + std::to_string(display_x) + "," + std::to_string(display_y) + ")",
                   cv::Point(cx + offset, cy - offset),
                   font_scale, text_thickness, color);
        draw_label(image, "s=" + cv::format("%.2f", det.score),
                   cv::Point(cx + offset, lower_y),
                   font_scale, text_thickness, color);
    }
}

void draw_coordinate_grid(cv::Mat& image, int step) {
    if (image.empty() || step <= 0) return;

    const int rows = image.rows;
    const int cols = image.cols;
    const int min_side = std::min(rows, cols);
    const int grid_thickness = std::max(1, static_cast<int>(std::round(min_side * 0.0014)));
    const int axis_thickness = std::max(2, grid_thickness + 1);
    const double font_scale = std::max(0.5, min_side / 1700.0);
    const int text_thickness = std::max(1, grid_thickness);

    const cv::Scalar grid_color(160, 160, 160);
    const cv::Scalar axis_color(0, 215, 255);

    cv::Mat overlay = image.clone();

    for (int x = 0; x < cols; x += step) {
        cv::line(overlay, cv::Point(x, 0), cv::Point(x, rows - 1),
                 grid_color, grid_thickness, cv::LINE_AA);
    }
    for (int y_value = 0; y_value < rows; y_value += step) {
        int y = rows - 1 - y_value;
        cv::line(overlay, cv::Point(0, y), cv::Point(cols - 1, y),
                 grid_color, grid_thickness, cv::LINE_AA);
    }

    cv::addWeighted(overlay, 0.18, image, 0.82, 0.0, image);

    cv::line(image, cv::Point(0, rows - 1), cv::Point(cols - 1, rows - 1),
             axis_color, axis_thickness, cv::LINE_AA);
    cv::line(image, cv::Point(0, rows - 1), cv::Point(0, 0),
             axis_color, axis_thickness, cv::LINE_AA);

    const int x_lines = std::max(1, cols / step);
    const int y_lines = std::max(1, rows / step);
    const int x_label_stride = std::max(1, static_cast<int>(std::ceil(static_cast<double>(x_lines) / 10.0)));
    const int y_label_stride = std::max(1, static_cast<int>(std::ceil(static_cast<double>(y_lines) / 8.0)));

    draw_label(image, "(0,0)", cv::Point(6, rows - 8), font_scale, text_thickness, axis_color);
    draw_label(image, "X", cv::Point(cols - 20, rows - 8), font_scale, text_thickness, axis_color);
    draw_label(image, "Y", cv::Point(6, 20), font_scale, text_thickness, axis_color);

    for (int idx = 1, x = step; x < cols; x += step, ++idx) {
        if ((idx % x_label_stride) != 0 && x + step < cols) continue;
        draw_label(image, std::to_string(x), cv::Point(x + 4, rows - 8),
                   font_scale, text_thickness, axis_color);
    }

    for (int idx = 1, y_value = step; y_value < rows; y_value += step, ++idx) {
        if ((idx % y_label_stride) != 0 && y_value + step < rows) continue;
        int y = rows - 1 - y_value;
        draw_label(image, std::to_string(y_value), cv::Point(6, y - 4),
                   font_scale, text_thickness, axis_color);
    }
}

void capture_thread(BoundedQueue<cv::Mat>& raw_queue, const StreamConfig& config) {
    if (!configure_high_res_pipeline(config)) {
        g_running = false;
        raw_queue.close();
        return;
    }

    V4L2Capture capture;
    if (!capture.open_device(config.camera_index, config.frame_w, config.frame_h, config.fps)) {
        g_running = false;
        raw_queue.close();
        return;
    }

    std::cout << "[Capture] Started, device=" << capture.device_path()
              << " requested=" << config.frame_w << "x" << config.frame_h << "@" << config.fps
              << " actual=" << capture.width() << "x" << capture.height()
              << " fmt=" << capture.pixel_format_name()
              << " fps=" << capture.fps() << std::endl;

    bool warned_resize = false;

    while (g_running.load()) {
        cv::Mat frame;
        if (!capture.read_frame(frame) || frame.empty()) {
            std::cerr << "[Capture] Failed to read frame, stopping" << std::endl;
            g_running = false;
            break;
        }

        if (frame.cols != config.frame_w || frame.rows != config.frame_h) {
            if (!warned_resize) {
                std::cerr << "[Capture] Driver output " << frame.cols << "x" << frame.rows
                          << " differs from requested " << config.frame_w << "x" << config.frame_h
                          << ", resizing in software" << std::endl;
                warned_resize = true;
            }
            cv::resize(frame, frame, cv::Size(config.frame_w, config.frame_h), 0, 0, cv::INTER_LINEAR);
        }

        raw_queue.push(std::move(frame));
    }

    capture.close_device();
    raw_queue.close();
    std::cout << "[Capture] Stopped" << std::endl;
}

void detect_thread(BoundedQueue<cv::Mat>& raw_queue,
                   BoundedQueue<cv::Mat>& result_queue,
                   const std::string& model_path,
                   float score_threshold, int topk, int nms_kernel,
                   int grid_step) {
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
        }

        {
            std::lock_guard<std::mutex> lock(g_snapshot_mutex);
            g_latest_snapshot.frame_index = frame_count + 1;
            g_latest_snapshot.frame_w = frame.cols;
            g_latest_snapshot.frame_h = frame.rows;
            g_latest_snapshot.inference_enabled = inference_enabled;
            g_latest_snapshot.detections = dets;
        }

        if (g_grid_enabled.load()) {
            draw_coordinate_grid(frame, grid_step);
        }
        if (inference_enabled) {
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
    {
        std::lock_guard<std::mutex> lock(g_snapshot_mutex);
        g_latest_snapshot = DetectionSnapshot{};
    }

    StreamConfig config;
    switch (parse_args(argc, argv, config)) {
    case ParseResult::kOk:
        break;
    case ParseResult::kHelp:
        return 0;
    case ParseResult::kError:
        return 1;
    }

    g_grid_enabled = (config.grid_step > 0);

    std::cout << "========================================\n"
              << "Real-time Spot Detection Stream\n"
              << "Model      : " << config.model_path << "\n"
              << "Target     : " << config.ip << ":5000\n"
              << "Camera     : " << config.camera_index << "\n"
              << "Preset     : " << config.resolution_preset << "\n"
              << "Resolution : " << config.frame_w << "x" << config.frame_h << "\n"
              << "FPS        : " << config.fps << "\n"
              << "Grid       : " << (config.grid_step > 0
                                      ? (std::to_string(config.grid_step) + " px, origin=bottom-left")
                                      : std::string("off")) << "\n"
              << "Threshold  : " << config.score_threshold << "\n"
              << "TopK       : " << config.topk << "\n"
              << "NMS Kernel : " << config.nms_kernel << "\n"
              << "Input q + Enter to toggle inference\n"
              << "Input w + Enter to toggle coordinate grid\n"
              << "Input e + Enter to export current frame spot coordinates\n"
              << "Press Ctrl+C to exit\n"
              << "========================================\n" << std::endl;

    BoundedQueue<cv::Mat> raw_queue(3);
    BoundedQueue<cv::Mat> result_queue(3);

    std::thread t_capture(capture_thread,
                          std::ref(raw_queue), std::cref(config));
    std::thread t_detect(detect_thread,
                         std::ref(raw_queue),
                         std::ref(result_queue),
                         std::cref(config.model_path),
                         config.score_threshold, config.topk, config.nms_kernel,
                         config.grid_step);
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
