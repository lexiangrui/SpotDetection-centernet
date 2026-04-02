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

### 1. 安装 RKNN Toolkit2（在开发机上）

RKNN Toolkit2 负责将 ONNX 模型转换为 RKNN 格式。仅需在 x86 开发机上安装，无需在板卡上运行。

```bash
# Python 3.8 ~ 3.11
pip install rknn-toolkit2 --extra-index-url https://download.rockchip.com/rknn/rknn-toolkit2/latest/
```

### 2. 导出 RKNN 模型（在开发机上）

支持两种量化模式：

| 模式 | 量化 | NPU 支持 | 预处理 | 推荐场景 |
| --- | --- | --- | --- | --- |
| **INT8** | INT8 量化 | 是（推荐） | uint8 输入，mean/std 由 NPU 融合 | 生产部署 |
| **FP32** | 不量化 | 否（CPU/GPU） | float32 输入，需 CPU 归一化 | 精度对比、调试 |

```bash
# 方式一：INT8 量化（模型用于 NPU 推理）
python scripts/export_rknn.py \
    --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
    --output deploy/model/spot_centernet_int8.rknn \
    --quantize int8

# 方式二：FP32 不量化
python scripts/export_rknn.py \
    --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
    --output deploy/model/spot_centernet_fp32.rknn \
    --quantize fp32

# 复用已有 ONNX（跳过导出步骤）
python scripts/export_rknn.py \
    --onnx models/spot_centernet_mobilenetv3_focal/best.onnx \
    --output deploy/model/spot_centernet_int8.rknn
```

INT8 量化需要标定数据集，脚本自动从 `splits/val.txt` 读取图片进行预处理并缓存到 `.calib_cache/calib_640x640.npy`。可加 `--reuse-calib` 复用缓存。

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--platform` | `rk3576` | 目标平台，如 `rk3576`、`rk3588`、`rk356x` |
| `--calib-size` | `100` | 标定图片数量（越多精度越好，越慢） |
| `--calib-split` | `splits/val.txt` | 标定图片来源 |
| `--photos-dir` | `photos` | 图片目录 |
| `--batch-size` | `1` | ONNX batch size |

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
./spot_stream [--model <path>] [--ip <addr>] [--camera <index>] [--threshold <float>] \
              [--topk <int>] [--nms-kernel <int>] [--video-mode <low|medium|high>] \
              [--fps <int>] [--width <int>] [--height <int>] [--grid-step <int>]
```

示例：

```bash
# 使用默认参数
./spot_stream

# 指定模型和目标 IP
./spot_stream --model ./model/spot_centernet.rknn --ip 192.168.99.230

# 指定摄像头、检测参数、分辨率预设和帧率
./spot_stream --camera 22 --threshold 0.1 --topk 256 --nms-kernel 9 --video-mode low --fps 30

# 使用板卡原生中分辨率实时流
./spot_stream --video-mode medium --fps 30 --grid-step 200

# 使用板卡最高原生分辨率实时流
./spot_stream --video-mode high --fps 15 --grid-step 200

# 查看帮助
./spot_stream --help
```

默认参数：


| 参数              | 默认值                           |
| --------------- | ----------------------------- |
| model_path      | `./model/spot_centernet.rknn` |
| target_ip       | `192.168.99.230`              |
| camera_index    | `22`                          |
| score_threshold | `0.1`                         |
| topk            | `256`                         |
| nms_kernel      | `9`                           |
| video_mode      | `low`                         |
| fps             | `30`                          |
| grid_step       | `100`                         |


分辨率预设选项：


| `video_mode` | 分辨率           |
| ------------ | ------------- |
| `low`        | `1920 x 1080` |
| `medium`     | `2112 x 1568` |
| `high`       | `4224 x 3136` |


参数说明：


| 参数             | 说明                                         |
| -------------- | ------------------------------------------ |
| `--model`      | `.rknn` 模型路径                               |
| `--ip`         | UDP 推流目标 IP，端口固定为 `5000`                   |
| `--camera`     | 摄像头索引                                      |
| `--threshold`  | 检测置信度阈值                                    |
| `--topk`       | NMS 前保留的候选点数量                              |
| `--nms-kernel` | NMS 核大小，要求为正奇数                             |
| `--video-mode` | 预设分辨率，支持 `low`、`medium` 和 `high`         |
| `--fps`        | 采集帧率，默认 `30`，不再与分辨率预设绑定                    |
| `--width`      | 自定义采集宽度，需要与 `--height` 一起提供                |
| `--height`     | 自定义采集高度，需要与 `--width` 一起提供                 |
| `--grid-step`  | 坐标网格间距，单位像素；默认 `100`，即默认开启网格，设置为 `0` 可关闭网格 |
| `--help`       | 打印完整帮助信息                                   |


也支持自定义采集尺寸，帧率仍通过 `--fps` 单独指定：

```bash
./spot_stream --width 2112 --height 1568 --fps 15
```

自定义尺寸时，需要同时提供 `--width` 和 `--height`；未显式指定 `--fps` 时默认使用 `30`。

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

- 采集线程按配置的分辨率预设或自定义尺寸、以及独立的 `fps` 从摄像头读取帧，放入 `raw_queue`
- 检测线程从 `raw_queue` 取帧，先叠加以左下角为原点的坐标网格，再运行 RKNN 模型检测光斑并绘制标记，放入 `result_queue`
- 推流线程从 `result_queue` 取帧，通过 GStreamer 编码为 H.264 并以 RTP/UDP 发送到目标 IP:5000
- 运行中输入 `q` 并回车，可切换推理开关；关闭推理时仍继续推流原始视频
- 运行中输入 `w` 并回车，可切换坐标网格开关；默认开启

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
