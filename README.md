# 光斑定位系统（CenterNet / ResNet18）

## 1. 项目简介

本项目实现了一套基于 CenterNet 的单类点目标检测系统，用于定位图像或视频中的光斑中心。模型输出：

- `heatmap`：1 通道中心点热力图
- `reg`：2 通道亚像素偏移

解码阶段对热力图做局部极大值抑制和 Top-K 选择，再结合 `reg` 回归原图坐标，输出 `(x, y, score)`。

当前代码固定使用 `resnet18` backbone。

支持 4 种热力图损失：

- `focal`
- `mse`
- `bce`
- `kl`

## 2. 当前网络结构

这条分支当前是固定的 `resnet18 + 3 stage conv + deconv decoder` 结构。

对应代码：

- [model.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/model.py)
- [neck.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/neck.py)
- [backbones](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/backbones)

检测流程：

```text
输入图像
  -> letterbox (resize + pad)
  -> resnet18 backbone (stride 4/8/16/32 多尺度特征)
  -> decoder
  -> CenterNet head
       |- heatmap: [B, 1, H/4, W/4]
       |- reg:     [B, 2, H/4, W/4]
  -> sigmoid + local NMS + Top-K
  -> 映射回原图坐标
```

## 3. 项目结构

```text
光斑定位-centernet/
├── configs/
│   └── spot_centernet.yaml
├── dataset/
│   ├── train/
│   └── val/
├── deploy/
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── include/
│   ├── model/
│   └── src/
├── docs/
│   └── 定位抖动修正说明.md
├── models/
│   └── spot_centernet_resnet18_focal/
├── outputs/
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── export_onnx.py
│   └── export_rknn.py
└── src/centernet_spot/
```

## 4. 环境安装

推荐 Python `3.10+`。

安装基础依赖：

```bash
pip install -e .
```

可选工具：

```bash
pip install labelme
pip install rknn-toolkit2 --extra-index-url https://download.rockchip.com/rknn/rknn-toolkit2/latest/
```

快速验证：

```bash
python -c "from centernet_spot import SpotCenterNet; print('import OK')"
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print('mps=', torch.backends.mps.is_available())"
```

## 5. 数据格式

默认训练数据位于 `dataset/`，按显式训练/验证目录组织。旧版加工数据已保留为 `dataset1/`：

```text
dataset/
├── train/
│   ├── xxx.jpg
│   └── xxx.json
└── val/
    ├── yyy.jpg
    └── yyy.json
```

标注约定：

- 光斑中心：`label = "spot"`，`shape_type = "point"`

当前训练数据加载逻辑会直接读取 `dataset/train/*.json` 和 `dataset/val/*.json` 中的 `spot(point)` 标注。

标注命令：

```bash
labelme dataset/train
labelme dataset/val
```

## 5. 训练

基本命令：

```bash
python scripts/train.py --config configs/spot_centernet.yaml
```

常用覆盖参数：

```bash
python scripts/train.py \
  --config configs/spot_centernet.yaml \
  --epochs 80 \
  --batch-size 8 \
  --save-dir models/spot_centernet_resnet18_focal_v2
```

当前训练脚本的行为：

- 自动刷新 `train/val split`
- 使用 `AdamW`
- 使用 `ReduceLROnPlateau` 监控 `val_loss`
- 用 `fitness = 0.7 * AP + 0.3 * F1` 选择 `best.pt`
- 支持 early stopping

训练输出目录默认是 `models/spot_centernet_resnet18_focal/`，其中会生成：

- `best.pt`
- `last.pt`
- `train.log`
- `metrics.jsonl`
- `metrics.csv`
- `metrics.json`
- `summary.json`
- `run_context.json`
- `loss_curve.png`
- `train_vis/epoch_XXX.jpg`

## 6. 推理

### 6.1 单图

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input photos/000001.jpg \
  --output outputs/infer_single
```

### 6.2 整个目录

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input dataset/val \
  --output outputs/infer_all
```

### 6.3 视频

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input video/capture4.mp4 \
  --output outputs/infer_video
```

推理输出说明：

- 图片输出：`{name}_vis.jpg` 和 `{name}.json`
- 视频输出：`{name}_vis.mp4` 或 `{name}_vis.avi`、`{name}_annotated.mp4` 或 `{name}_annotated.avi`，以及 `{name}.json`
- 可视化图是三栏拼图：原图 / 恢复到原尺寸的热力图 / 检测标注图

JSON 中每个检测结果会带：

- `x`
- `y`
- `score`
- `spot_id`

其中 `spot_id` 会按“从上到下、从左到右”重新编号。可视化里显示的坐标文字使用 `(x, H - y)`，这是为了和部署端统一成“左下角为原点”的显示习惯；JSON 原始 `x/y` 仍然是图像坐标。

## 7. ONNX 与 RKNN 导出

### 7.1 导出 ONNX

```bash
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output outputs/best.onnx
```

可选参数：

- `--config`：当 checkpoint 里没有嵌入配置时手动指定 YAML
- `--opset`：默认 `17`
- `--batch-size`：默认 `1`
- `--dynamic-batch`：开启动态 batch 维

### 7.2 导出 RKNN

INT8：

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output deploy/model/spot_centernet_resnet18_int8.rknn \
  --quantize int8
```

FP16：

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output deploy/model/spot_centernet_resnet18_fp.rknn \
  --quantize fp16
```

复用已有 ONNX：

```bash
python scripts/export_rknn.py \
  --onnx outputs/best.onnx \
  --output deploy/model/spot_centernet_resnet18_int8.rknn \
  --quantize int8
```

当前 `export_rknn.py` 会：

- 需要时先导出 ONNX
- 直接从 `dataset/val` 读取标定样本
- 做与部署一致的 RGB letterbox 预处理
- 在 `.calib_cache/` 下缓存标定 `.npy/.txt`
- 输出 `int8` 或 `fp16` RKNN 模型

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--platform` | `rk3576` | 目标平台 |
| `--calib-image-dir` | `dataset/val` | 标定图片目录 |
| `--calib-size` | `100` | 标定图片上限 |
| `--reuse-calib` | 关闭 | 复用已有标定缓存 |

## 8. 核心源码入口

| 路径 | 作用 |
| --- | --- |
| `src/centernet_spot/model.py` | 组装 `resnet18 + decoder + heads` |
| `src/centernet_spot/backbones/resnet18.py` | 固定 backbone |
| `src/centernet_spot/neck.py` | 解码器 |
| `src/centernet_spot/head.py` | `heatmap/reg` 头 |
| `src/centernet_spot/data.py` | 数据加载与监督目标 |
| `src/centernet_spot/decode.py` | NMS、Top-K、坐标反变换 |
| `src/centernet_spot/preprocessing.py` | Python 推理预处理 |
| `src/centernet_spot/transforms.py` | letterbox 与尺寸映射 |
| `src/centernet_spot/visualization.py` | 三栏可视化与损失曲线 |
| `deploy/src/spot_detector.cpp` | RKNN 端预处理、推理、后处理 |

## 9. 补充说明

- 当前分支已经不再保留多 backbone 切换逻辑，README 中所有命令都默认按 `resnet18` 路径写。
- 当前 Python 端与部署端都基于 `heatmap + reg` 双输出；如果你改模型输出头，训练、导出和部署都要一起改。
- 如果你在意视频中坐标抖动，可继续参考 `docs/定位抖动修正说明.md`。
