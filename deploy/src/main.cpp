#include "spot_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

static void draw_crosshair(cv::Mat& canvas, int cx, int cy,
                           int size, const cv::Scalar& color, int thickness) {
    cv::line(canvas, cv::Point(cx - size, cy), cv::Point(cx + size, cy),
             color, thickness, cv::LINE_AA);
    cv::line(canvas, cv::Point(cx, cy - size), cv::Point(cx, cy + size),
             color, thickness, cv::LINE_AA);
}

static std::string make_detection_label(const cv::Mat& image, const Detection& det) {
    int x = std::clamp(static_cast<int>(std::lround(det.x)), 0, image.cols);
    int y = std::clamp(static_cast<int>(std::lround(static_cast<double>(image.rows) - det.y)), 0, image.rows);
    char label[96];
    snprintf(label, sizeof(label), "x=%d y=%d s=%.2f", x, y, det.score);
    return std::string(label);
}

static void draw_label(cv::Mat& image, const std::string& label, cv::Point origin,
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

static void draw_detections(cv::Mat& image, const std::vector<Detection>& dets) {
    int min_side = std::min(image.rows, image.cols);
    int marker_size = std::max(6, static_cast<int>(std::round(min_side * 0.012)));
    int marker_thickness = std::max(1, static_cast<int>(std::round(marker_size / 4.5)));
    double font_scale = std::max(0.36, marker_size / 15.0);
    int text_thickness = std::max(1, marker_thickness);
    cv::Scalar color(0, 255, 0);

    for (const auto& det : dets) {
        int cx = static_cast<int>(std::round(det.x));
        int cy = static_cast<int>(std::round(det.y));
        draw_crosshair(image, cx, cy, marker_size, color, marker_thickness);
        draw_label(image, make_detection_label(image, det),
                   cv::Point(cx + marker_size + 4, cy - std::max(marker_size / 2, 4)),
                   font_scale, text_thickness, color);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rknn_model> <image_path> [score_threshold] [topk] [nms_kernel]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    float score_threshold = (argc > 3) ? std::stof(argv[3]) : 0.3f;
    int topk = (argc > 4) ? std::stoi(argv[4]) : 256;
    int nms_kernel = (argc > 5) ? std::stoi(argv[5]) : 5;

    SpotDetector detector;
    int ret = detector.init(model_path);
    if (ret != 0) {
        fprintf(stderr, "[ERR] Failed to init detector\n");
        return 1;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        fprintf(stderr, "[ERR] Failed to read image: %s\n", image_path.c_str());
        return 1;
    }

    auto dets = detector.detect(image, score_threshold, topk, nms_kernel);
    printf("[INFO] Detected %zu spots in %s\n", dets.size(), image_path.c_str());

    for (size_t i = 0; i < dets.size(); ++i) {
        float display_x = dets[i].x;
        float display_y = static_cast<float>(image.rows) - dets[i].y;
        printf("  [%zu] x=%.2f y=%.2f score=%.4f\n", i, display_x, display_y, dets[i].score);
    }

    draw_detections(image, dets);

    std::string output_path = "result.jpg";
    cv::imwrite(output_path, image);
    printf("[INFO] Saved: %s\n", output_path.c_str());

    return 0;
}
