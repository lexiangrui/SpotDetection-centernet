# RK3576 部署指南

## 1. 概述

本目录包含 RKNN 端 C++ 推理代码，提供两种入口：

- `spot_detect`：单张图片推理
- `spot_stream`：摄像头采集 + 检测 + UDP 推流

目录结构：

```text
deploy/
├── CMakeLists.txt
├── README.md
├── include/
│   └── spot_detector.h
├── src/
│   ├── spot_detector.cpp
│   ├── main.cpp
│   └── main_stream.cpp
└── model/
```

## 2. 当前端侧代码的前提

当前 [spot_detector.cpp](/Users/lexiangrui/Desktop/光斑定位-centernet/deploy/src/spot_detector.cpp) 对模型输出有明确假设：

- 只有 2 个输出：`heatmap`、`reg`
- 输出 stride 固定为 `4`
- 当输入是 `640 x 640` 时，输出默认为 `160 x 160`

也就是说，端侧代码目前适配的是标准 CenterNet 风格 `H/4 x W/4` 输出，不适配全分辨率输出模型。

预处理使用 letterbox：

- 等比例缩放
- padding 到模型输入尺寸

后处理再把坐标映射回原图。

## 3. 当前代码和部署的关系

主仓库当前支持三条模型路径：

- `resnet18 + CenterNet-style DCN/deconv decoder`
- `dla34 + DLAUp/IDAUp decoder`
- `mobilenetv3_large + BiFPN-style decoder`

其中：

- `resnet18` 和 `dla34` decoder 依赖 `torchvision.ops.DeformConv2d`
- `mobilenetv3_large` decoder 不依赖 DCN

这会直接影响部署：

- `resnet18/dla34` 在 ONNX/RKNN 导出时更容易失败
- 如果你的目标是稳定导出到 RKNN，当前代码里更推荐使用 `mobilenetv3_large`

## 4. 开发机上准备 RKNN 模型

安装：

```bash
pip install rknn-toolkit2 --extra-index-url https://download.rockchip.com/rknn/rknn-toolkit2/latest/
```

### 4.1 导出 ONNX

如果你当前训练的是部署友好的 `mobilenetv3_large` 路径：

```bash
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --output outputs/best.onnx
```

### 4.2 导出 RKNN

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --output deploy/model/spot_centernet_int8.rknn \
  --quantize int8
```

或不量化：

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --output deploy/model/spot_centernet_fp32.rknn \
  --quantize fp32
```

也可以复用已有 ONNX：

```bash
python scripts/export_rknn.py \
  --onnx outputs/best.onnx \
  --output deploy/model/spot_centernet_int8.rknn
```

`export_rknn.py` 当前会自动：

- 从 `splits/val.txt` 读取标定样本
- 做统一预处理
- 在 `.calib_cache/` 下缓存 `.npy/.txt`
- 再调用 RKNN Toolkit 转换

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--platform` | `rk3576` | 目标平台 |
| `--calib-size` | `100` | 标定图数量 |
| `--calib-split` | `splits/val.txt` | 标定样本来源 |
| `--photos-dir` | `photos` | 图片目录 |
| `--reuse-calib` | 关闭 | 复用缓存标定数据 |

## 5. 将文件传到板卡

需要复制：

- `deploy/` 整个目录
- `.rknn` 模型文件
- 测试图片或视频

## 6. 编译

依赖：

- RKNN Runtime
- OpenCV 4.x
- CMake 3.10+
- C++17 编译器
- GStreamer 1.0 + `gstreamer-app-1.0`（仅 `spot_stream`）

编译：

```bash
cd deploy
mkdir -p build && cd build

cmake .. \
  -G Ninja \
  -DRKNN_API_PATH=/usr/local \
  -DCMAKE_BUILD_TYPE=Release

ninja
```

## 7. 运行

### 7.1 单张图片

```bash
./spot_detect <rknn_model> <image_path> [score_threshold] [topk] [nms_kernel]
```

示例：

```bash
./spot_detect ../model/spot_centernet_int8.rknn test.jpg
./spot_detect ../model/spot_centernet_int8.rknn test.jpg 0.3 256 5
```

输出：

- 终端打印检测坐标
- 保存 `result.jpg`

### 7.2 实时推流

```bash
./spot_stream [--model <path>] [--ip <addr>] [--camera <index>] [--threshold <float>] \
              [--topk <int>] [--nms-kernel <int>] [--video-mode <low|medium|high>] \
              [--fps <int>] [--width <int>] [--height <int>] [--grid-step <int>]
```

示例：

```bash
./spot_stream
./spot_stream --model ./model/spot_centernet_int8.rknn --ip 192.168.99.230
./spot_stream --camera 22 --threshold 0.1 --topk 256 --nms-kernel 9 --video-mode low --fps 30
```

默认参数：

| 参数 | 默认值 |
| --- | --- |
| `model_path` | `./model/spot_centernet.rknn` |
| `target_ip` | `192.168.99.230` |
| `camera_index` | `22` |
| `score_threshold` | `0.1` |
| `topk` | `256` |
| `nms_kernel` | `9` |
| `video_mode` | `low` |
| `fps` | `30` |
| `grid_step` | `100` |

分辨率预设：

| `video_mode` | 分辨率 |
| --- | --- |
| `low` | `1920 x 1080` |
| `medium` | `2112 x 1568` |
| `high` | `4224 x 3136` |

## 8. 模型接口要求

当前 C++ 端要求模型满足：

- 输入：RGB letterbox 到固定尺寸
- 输出：
  - `heatmap [1,1,160,160]`
  - `reg [1,2,160,160]`
- 坐标语义：CenterNet `heatmap + reg`

如果你改了这些任一项，`deploy/src/spot_detector.cpp` 也要一起改。

## 9. 使用建议

如果你的目标是“这条分支直接导出 RKNN 并部署”，优先建议：

```yaml
model:
  backbone: mobilenetv3_large
  decoder_channels: 96
```

原因很直接：

- 端侧代码只关心 `heatmap/reg` 和 stride 4
- `mobilenetv3_large` 路径没有 DCN
- `resnet18/dla34` 当前更容易卡在 ONNX/RKNN 导出阶段
