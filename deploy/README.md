# RK3576 部署指南

## 概述

本目录包含在 RK3576 上运行光斑检测模型的 C++ 推理代码，基于 RKNN Runtime API。

## 目录结构

```text
deploy/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── include/
│   └── spot_detector.h         # 检测器头文件
├── src/
│   ├── spot_detector.cpp       # 检测器实现（预处理 + 推理 + 后处理）
│   └── main.cpp                # 命令行入口
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
- 测试图片

## 编译（在 RK3576 板卡上）

### 依赖

- RKNN Runtime（`librknnrt.so` + `rknn_api.h`）
- OpenCV 4.x
- CMake 3.10+
- GCC/G++ 支持 C++17

### 编译步骤

```bash
cd deploy
mkdir -p build && cd build

cmake .. \
  -DRKNN_API_PATH=/usr/local \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

如果 RKNN API 安装在其他路径，修改 `-DRKNN_API_PATH`。

## 运行

```bash
./spot_detect <rknn_model> <image_path> [score_threshold] [topk] [nms_kernel]
```

### 示例

```bash
# 使用默认参数 (threshold=0.6, topk=256, nms_kernel=5)
./spot_detect ../model/spot_centernet.rknn test.jpg

# 自定义参数
./spot_detect ../model/spot_centernet.rknn test.jpg 0.5 256 5
```

### 输出

- 终端打印每个检测到的光斑坐标和置信度
- 保存标注结果图 `result.jpg`

## 模型参数

| 参数 | 值 |
| --- | --- |
| 骨干网络 | MobileNetV3-Large |
| 输入尺寸 | 640 × 640 (RGB) |
| 输出尺寸 | 160 × 160 |
| 归一化 | ImageNet mean/std |
| 输出 | heatmap [1,1,160,160] + reg [1,2,160,160] |

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
