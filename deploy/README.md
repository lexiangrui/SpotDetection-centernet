# RK3576 部署指南

## 概述

本目录包含在 RK3576 上运行光斑检测模型的 C++ 推理代码，基于 RKNN Runtime API。

提供两种运行模式：

- **spot_detect** — 单张图片推理，用于离线测试
- **spot_stream** — 实时摄像头采集 + 检测 + GStreamer UDP 推流

## 目录结构

```text
deploy/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── include/
│   └── spot_detector.h         # 检测器头文件
├── src/
│   ├── spot_detector.cpp       # 检测器实现（预处理 + 推理 + 后处理）
│   ├── main.cpp                # 单张图片推理入口
│   └── main_stream.cpp         # 实时推流入口（三线程流水线）
└── model/                      # 存放 .rknn 模型文件
```

## 准备工作

### 1. 导出 ONNX（在开发机上）

```bash
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --output models/spot_centernet_mobilenetv3_focal/best.onnx
```

### 2. 转换为 RKNN（在开发机上，需要 rknn-toolkit2）

```bash
python scripts/export_rknn.py \
  --onnx models/spot_centernet_mobilenetv3_focal/best.onnx \
  --output deploy/model/spot_centernet.rknn \
  --target-platform rk3576
```

如需 INT8 量化：

```bash
# 先制备量化数据集
python scripts/make_rknn_dataset.py \
  --photos photos \
  --out-dir outputs/rknn_dataset \
  --dataset-txt outputs/rknn_dataset.txt

# 再转换并量化
python scripts/export_rknn.py \
  --onnx models/spot_centernet_mobilenetv3_focal/best.onnx \
  --output deploy/model/spot_centernet_int8.rknn \
  --target-platform rk3576 \
  --quantize \
  --dataset outputs/rknn_dataset.txt
```

### 3. 将文件传输到 RK3576 板卡

将以下文件复制到板卡：

- `deploy/` 整个目录
- 测试图片（如使用 `spot_detect`）

## 编译（在 RK3576 板卡上）

### 依赖

- RKNN Runtime（`librknnrt.so` + `rknn_api.h`）
- OpenCV 4.x
- CMake 3.10+
- GCC/G++支持 C++17
- GStreamer 1.0 + gstreamer-app-1.0（仅 `spot_stream` 需要）

### 编译步骤

```bash
cd deploy
mkdir -p build && cd build

cmake .. \
  -G Ninja \
  -DRKNN_API_PATH=/usr/local \
  -DCMAKE_BUILD_TYPE=Release

ninja
```

如果 RKNN API 安装在其他路径，修改 `-DRKNN_API_PATH`。

如果未安装 GStreamer，CMake 会跳过 `spot_stream` 的构建，`spot_detect` 仍可正常编译。

## 运行

### 模式一：单张图片推理 (spot_detect)

```bash
./spot_detect <rknn_model> <image_path> [score_threshold] [topk] [nms_kernel]
```

示例：

```bash
./spot_detect ../model/spot_centernet.rknn test.jpg
./spot_detect ../model/spot_centernet.rknn test.jpg 0.5 256 5
```

输出：终端打印坐标 + 保存 `result.jpg`。

### 模式二：实时推流 (spot_stream)

```bash
./spot_stream [model_path] [target_ip] [camera_index] [score_threshold] [topk] [nms_kernel]
```

示例：

```bash
# 使用默认参数
./spot_stream ../model/spot_centernet.rknn 192.168.99.230

# 指定摄像头和检测参数
./spot_stream ../model/spot_centernet.rknn 192.168.1.100 0 0.5 256 5
```

默认参数：


| 参数              | 默认值                           |
| --------------- | ----------------------------- |
| model_path      | `./model/spot_centernet.rknn` |
| target_ip       | `192.168.99.230`              |
| camera_index    | `22`                          |
| score_threshold | `0.6`                         |
| topk            | `256`                         |
| nms_kernel      | `5`                           |


#### 数据流水线

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ capture_thread│────▶│ detect_thread │────▶│  stream_thread   │
│ (摄像头采集)  │     │ (RKNN 推理)   │     │ (GStreamer 推流)  │
└──────────────┘     └──────────────┘     └──────────────────┘
        │                     │                      │
        ▼                     ▼                      ▼
  cv::VideoCapture      SpotDetector          appsrc → mpph264enc
  raw_queue (max 3)     result_queue (max 3)  → rtph264pay → udpsink
```

- 采集线程以 1280×720@30fps 从摄像头读取帧，放入 `raw_queue`
- 检测线程从 `raw_queue` 取帧，运行 RKNN 模型检测光斑并绘制标记，放入 `result_queue`
- 推流线程从 `result_queue` 取帧，通过 GStreamer 编码为 H.264 并以 RTP/UDP 发送到目标 IP:5000

#### 接收端查看

Linux：

```bash
gst-launch-1.0 -e -v \
udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! \
rtpjitterbuffer ! rtph264depay ! h264parse config-interval=-1 ! \
mp4mux ! filesink location=/home/yuexr/data/capture.mp4
```

或使用 ffplay：

```bash
ffplay -protocol_whitelist file,rtp,udp -i stream.sdp
```

`stream.sdp` 内容：

```
v=0
m=video 5000 RTP/AVP 96
c=IN IP4 <板卡IP>
a=rtpmap:96 H264/90000
```

## 模型参数


| 参数   | 值                                         |
| ---- | ----------------------------------------- |
| 骨干网络 | MobileNetV3-Large                         |
| 输入尺寸 | 640 × 640 (RGB)                           |
| 输出尺寸 | 160 × 160                                 |
| 归一化  | ImageNet mean/std                         |
| 输出   | heatmap [1,1,160,160] + reg [1,2,160,160] |


## 集成到自己的项目

`SpotDetector` 类可作为库使用：

```cpp
#include "spot_detector.h"

SpotDetector detector;
detector.init("model.rknn", 640, 640);

cv::Mat image = cv::imread("photo.jpg");
auto dets = detector.detect(image, 0.6f, 256, 5);

for (const auto& d : dets) {
    printf("x=%.2f y=%.2f score=%.4f\n", d.x, d.y, d.score);
}
```

CMake 集成：

```cmake
add_subdirectory(deploy)
target_link_libraries(your_target PRIVATE spot_detector)
```

