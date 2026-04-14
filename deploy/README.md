# RK3576 部署说明

## 1. 目录概览

`deploy/` 目录是当前仓库的 RKNN C++ 部署端，包含两个可执行入口：

- `spot_detect`：单张图片推理
- `spot_stream`：摄像头采集、板端检测、UDP 推流

目录结构：

```text
deploy/
├── CMakeLists.txt
├── README.md
├── include/
│   └── spot_detector.h
├── model/
│   ├── spot_centernet_resnet18_fp.rknn
│   └── spot_centernet_resnet18_int8.rknn
└── src/
    ├── main.cpp
    ├── main_stream.cpp
    └── spot_detector.cpp
```

## 2. 当前部署端和模型的对应关系

当前部署端是按仓库里的这一套模型接口：

- 输入：RGB letterbox 到模型输入尺寸
- 输出 1：`heatmap`
- 输出 2：`reg`
- 输出分辨率：`H/4 x W/4`

部署端会在运行时主动查询 RKNN 实际张量尺寸，但当前主配置 `configs/spot_centernet.yaml` 对应的默认尺寸仍然是：

- 输入：`640 x 480`
- 输出：`160 x 120`

`spot_detector.cpp` 当前已经支持两条模型输入路径：

- `INT8` 模型：直接喂 `uint8 RGB` letterbox 图，均值方差由 RKNN 侧融合
- `FP16/FP32` 模型：先在 CPU 上做 ImageNet mean/std 归一化，再喂浮点输入

这也是为什么 `deploy/model/` 里现在同时放了：

- `spot_centernet_resnet18_int8.rknn`
- `spot_centernet_resnet18_fp.rknn`

## 3. 开发机导出 RKNN

在仓库根目录执行。

安装 RKNN Toolkit：

```bash
pip install rknn-toolkit2 --extra-index-url https://download.rockchip.com/rknn/rknn-toolkit2/latest/
```

### 3.1 导出 ONNX

```bash
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output outputs/best.onnx
```

### 3.2 导出 INT8 RKNN

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output deploy/model/spot_centernet_resnet18_int8.rknn \
  --quantize int8
```

### 3.3 导出 FP16 RKNN

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output deploy/model/spot_centernet_resnet18_fp.rknn \
  --quantize fp16
```

也可以复用已有 ONNX：

```bash
python scripts/export_rknn.py \
  --onnx outputs/best.onnx \
  --output deploy/model/spot_centernet_resnet18_int8.rknn \
  --quantize int8
```

当前 `export_rknn.py` 的实际行为：

- 需要时先导出 ONNX
- `int8` 模式从 `splits/val.txt` 读取标定样本
- 标定数据会缓存到 `.calib_cache/`
- `fp16` 模式跳过标定

## 4. 板端准备

需要带到板卡上的最小集合：

- `deploy/` 整个目录
- `.rknn` 模型文件
- 测试图片，或可访问的摄像头设备

板端依赖：

- RKNN Runtime
- OpenCV 4.x
- CMake 3.10+
- C++17 编译器
- GStreamer 1.0 与 `gstreamer-app-1.0`（只在构建 `spot_stream` 时需要）

## 5. 编译

```bash
cd deploy
mkdir -p build
cd build

cmake .. \
  -G Ninja \
  -DRKNN_API_PATH=/usr/local \
  -DCMAKE_BUILD_TYPE=Release

ninja
```

当前 `CMakeLists.txt` 的行为：

- 总是构建静态库 `spot_detector`
- 总是构建 `spot_detect`
- 只有检测到 GStreamer 时才构建 `spot_stream`

如果找不到 RKNN Runtime，可以显式传：

```bash
cmake .. -DRKNN_API_PATH=/path/to/rknn/runtime
```

## 6. 运行单图推理

命令：

```bash
./spot_detect <rknn_model> <image_path> [score_threshold] [topk] [nms_kernel]
```

示例：

```bash
./spot_detect ../model/spot_centernet_resnet18_int8.rknn test.jpg
./spot_detect ../model/spot_centernet_resnet18_fp.rknn test.jpg 0.3 256 5
```

输出行为：

- 终端打印检测数量和每个光斑坐标
- 生成 `result.jpg`

坐标显示习惯：

- 检测内部仍按图像坐标处理
- 终端与画面上显示的 `y` 会转换成 `H - y`
- 编号按“从上到下、从左到右”排序

## 7. 运行实时推流

命令：

```bash
./spot_stream [--model <path>] [--ip <addr>] [--camera <index>] [--threshold <float>] \
              [--topk <int>] [--nms-kernel <int>] [--grid-step <int>] \
              [--video-mode <low|medium|high>] [--width <int>] [--height <int>] \
              [--fps <int>]
```

示例：

```bash
./spot_stream
./spot_stream --model ./model/spot_centernet_resnet18_int8.rknn --ip 192.168.99.230
./spot_stream --video-mode medium --fps 30 --grid-step 200
./spot_stream --video-mode high --fps 15 --grid-step 200
./spot_stream --width 2112 --height 1568 --fps 15
```

默认参数：

| 参数 | 默认值 |
| --- | --- |
| `model_path` | `./model/spot_centernet_resnet18_int8.rknn` |
| `target_ip` | `192.168.99.230` |
| `camera_index` | `22` |
| `score_threshold` | `0.3` |
| `topk` | `256` |
| `nms_kernel` | `5` |
| `grid_step` | `100` |
| `video_mode` | `low` |
| `fps` | `30` |

分辨率预设：

| 预设 | 分辨率 |
| --- | --- |
| `low` | `1920 x 1080` |
| `medium` | `2112 x 1568` |
| `high` | `4224 x 3136` |

运行时控制：

- 输入 `q` 回车：开关检测，视频流不断
- 输入 `w` 回车：开关坐标网格
- 输入 `e` 回车：导出当前帧坐标到 `outputs/spot_coords_frame_<frame>_<timestamp>.txt`

UDP 推流端口固定为 `5000`。

## 8. 高分辨模式说明

`--video-mode high` 不是单纯把 `/dev/video22` 的宽高改成 `4224 x 3136`。

当前 `main_stream.cpp` 已经内置了板卡专用的 subdev 预配置逻辑，会在打开视频节点前先尝试切换：

- `/dev/v4l-subdev2`：sensor source
- `/dev/v4l-subdev1`：dphy sink
- `/dev/v4l-subdev0`：csi sink
- `/dev/v4l-subdev5`：cif source
- `/dev/v4l-subdev4`：isp sink / mainpath source

也就是说，高分辨模式的真实流程是：

1. 先切 sensor / CSI / ISP 链路格式
2. 等链路稳定
3. 再打开 `/dev/video22`
4. 再申请 buffer 并开始采集
