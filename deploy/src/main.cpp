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

static void draw_label(cv::Mat& image, const std::string& label, cv::Point origin,
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

static void draw_detections(cv::Mat& image, const std::vector<Detection>& dets) {
    int min_side = std::min(image.rows, image.cols);
    int marker_size = std::max(6, static_cast<int>(std::round(min_side * 0.012)));
    int marker_thickness = std::max(1, static_cast<int>(std::round(marker_size / 4.5)));
    double font_scale = std::max(0.24, marker_size / 21.0);
    int text_thickness = std::min(3, std::max(1, static_cast<int>(std::round(min_side / 1200.0))));
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

    for (const auto& det : dets) {
        float display_x = det.x;
        float display_y = static_cast<float>(image.rows) - det.y;
        printf("  [%d] x=%.2f y=%.2f score=%.4f\n", det.id, display_x, display_y, det.score);
    }

    draw_detections(image, dets);

    std::string output_path = "result.jpg";
    cv::imwrite(output_path, image);
    printf("[INFO] Saved: %s\n", output_path.c_str());

    return 0;
}
