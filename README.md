# 光斑质心定位系统

## 1. 项目简介

本项目实现了一套基于 CenterNet 的单类点目标检测系统，用于定位图像或视频中的光斑中心。模型输出：

- `heatmap`：1 通道中心点热力图
- `reg`：2 通道亚像素偏移

解码阶段对热力图做局部极大值抑制和 Top-K 选择，再结合 `reg` 回归原图坐标，输出 `(x, y, score)`。

当前代码支持 3 种骨干网络：

- `resnet18`
- `dla34`
- `mobilenetv3_large`

支持 4 种热力图损失：

- `focal`
- `mse`
- `bce`
- `kl`

## 2. 当前网络结构

这条分支当前不是“统一 FPN”结构，而是“多骨干 + 各自 decoder”：

- `resnet18`
  使用 `3 stage conv + deconv` decoder
- `dla34`
  使用 `DLAUp + IDAUp` decoder
- `mobilenetv3_large`
  使用轻量 `BiFPN-style` decoder

对应代码：

- [model.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/model.py)
- [neck.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/neck.py)
- [backbones](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/backbones)

检测流程：

```text
输入图像
  -> letterbox (resize + pad)
  -> backbone (stride 4/8/16/32 多尺度特征)
  -> decoder (按 backbone 分支实现)
  -> CenterNet head
       |- heatmap: [B, 1, H/4, W/4]
       |- reg:     [B, 2, H/4, W/4]
  -> sigmoid + local NMS + Top-K
  -> 映射回原图坐标
```

## 3. 项目结构

```text
光斑定位-centernet/
├── pyproject.toml
├── README.md
├── configs/
│   └── spot_centernet.yaml
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── export_onnx.py
│   └── export_rknn.py
├── src/centernet_spot/
│   ├── __init__.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── utils.py
│   │   ├── dla34.py
│   │   ├── resnet18.py
│   │   └── mobilenetv3.py
│   ├── neck.py
│   ├── head.py
│   ├── model.py
│   ├── config.py
│   ├── data.py
│   ├── decode.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── preprocessing.py
│   ├── transforms.py
│   ├── split.py
│   ├── utils.py
│   └── visualization.py
├── photos/
├── labels_raw/
├── splits/
├── models/
└── outputs/
```

## 4. 环境配置

推荐 Python `3.10+`。

安装：

```bash
cd 光斑定位-centernet
pip install -e .
```

可选依赖：

```bash
pip install labelme
pip install rknn-toolkit2
```

验证：

```bash
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print('mps=', torch.backends.mps.is_available())"
python -c "from centernet_spot import SpotCenterNet; print('import OK')"
```

## 5. 数据格式

图片放在 `photos/`，标注放在 `labels_raw/`。

使用 Labelme 标注：

```bash
labelme photos/ --output labels_raw/
```

每个光斑中心使用 Labelme：

- `label = "spot"`
- `shape_type = "point"`

每张图还需要一条代表性尺寸线段：

- `label = "spot_size"`
- `shape_type = "line"`

`train.py` 每次训练前会自动刷新 `splits/train.txt` 和 `splits/val.txt`。

## 6. 配置文件

配置文件：`configs/spot_centernet.yaml`

当前默认值：

| 字段 | 当前值 | 说明 |
| --- | --- | --- |
| `model.backbone` | `resnet18` | 当前默认骨干 |
| `model.decoder_channels` | `96` | decoder 通道数；仅 `mobilenetv3_large` 路径实际使用 |
| `model.head_channels` | `48` | 检测头通道数 |
| `model.backbone_kwargs.pretrained` | `true` | 加载预训练权重 |
| `model.backbone_kwargs.weights` | `default` | torchvision 默认权重 |
| `data.input_width` | `640` | 输入宽度 |
| `data.input_height` | `640` | 输入高度 |
| `data.down_ratio` | `4` | 输出 stride |
| `train.save_dir` | `models/spot_centernet_resnet18_focal` | 默认训练输出目录 |
| `train.heatmap_loss_type` | `focal` | 热力图损失 |
| `infer.score_threshold` | `0.3` | 推理阈值 |
| `infer.topk` | `256` | Top-K 数量 |
| `infer.nms_kernel` | `5` | 推理 NMS 核大小 |

输入输出关系：

- 输入默认 `640 x 640`
- 输出默认 `160 x 160`

输入预处理使用 letterbox，也就是等比例缩放加 padding：

- Python 训练/推理见 [transforms.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/transforms.py)
- RKNN 端预处理见 [spot_detector.cpp](/Users/lexiangrui/Desktop/光斑定位-centernet/deploy/src/spot_detector.cpp)

归一化逻辑：

- 若 `model.input_normalization` 有配置，直接使用配置值
- 若没显式配置且骨干使用预训练权重，自动回退到 ImageNet mean/std

对应实现见 [utils.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/utils.py)

## 7. 训练

基本命令：

```bash
python scripts/train.py --config configs/spot_centernet.yaml
```

可选参数：

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径 |
| `--epochs` | 覆盖训练轮数 |
| `--batch-size` | 覆盖批次大小 |
| `--save-dir` | 覆盖输出目录 |

训练输出：

- `best.pt`
- `last.pt`
- `metrics.json`
- `summary.json`
- `loss_curve.png`
- `train_vis/epoch_XXX.jpg`

## 8. 推理

单张图片：

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input photos/000001.jpg \
  --output outputs/infer_single
```

整个目录：

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input photos \
  --output outputs/infer_all
```

视频：

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input video/capture.mp4 \
  --output outputs/infer_video
```

输出：

- 图片：`{name}_vis.jpg` 和 `{name}.json`
- 视频：`{name}_vis.mp4/.avi` 和 `{name}.json`

## 9. ONNX 与 RKNN 导出

导出 ONNX：

```bash
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output outputs/best.onnx
```

导出 RKNN：

```bash
python scripts/export_rknn.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output outputs/best_int8.rknn \
  --quantize int8
```

`export_rknn.py` 当前会自动：

- 从 `splits/val.txt` 读取标定图片
- 做统一预处理
- 在 `.calib_cache/` 下缓存 `.npy/.txt`
- 再执行 RKNN 转换

当前仓库没有 `scripts/make_rknn_dataset.py`。

### 导出与训练说明

当前三条 decoder 路径都只使用标准卷积、深度可分离卷积和反卷积，不再依赖 `DeformConv2d`。

如果你的目标是更轻量的部署，当前代码里仍然更推荐改用：

```yaml
model:
  backbone: mobilenetv3_large
  decoder_channels: 96
```

因为 `mobilenetv3_large` 参数量和计算量通常更友好。

## 10. 调参建议

| 现象 | 建议 |
| --- | --- |
| 漏检较多 | 降低 `infer.score_threshold`；检查 `spot_size` 标定是否偏小 |
| 误检较多 | 提高 `infer.score_threshold`；增加训练样本 |
| 相邻光斑粘连 | 减小 `infer.nms_kernel`；检查 `spot_size` 标定线是否过长 |
| 训练不稳定 | 降低 `train.lr`；增大 `train.batch_size`；若使用 `mobilenetv3_large` 可调整 `decoder_channels` |
| 部署导出失败 | 先检查当前 checkpoint 和配置文件是否匹配，再检查 ONNX/RKNN 环境版本 |

## 11. 核心源码说明

| 文件 | 说明 |
| --- | --- |
| `backbones/` | backbone 注册表与各骨干实现 |
| `neck.py` | 各 backbone 对应 decoder 实现 |
| `head.py` | heatmap/reg 检测头 |
| `model.py` | backbone + decoder + head 组装 |
| `data.py` | 数据加载与监督目标生成 |
| `decode.py` | NMS、Top-K、坐标反变换 |
| `preprocessing.py` | 推理与 RKNN 共享预处理 |
| `transforms.py` | letterbox 与坐标映射 |
| `visualization.py` | 可视化与训练曲线导出 |
| `utils.py` | 设备、归一化、JSON/目录工具 |

## 12. 常见问题

### Q1：为什么 README 里不再写“统一 FPN”

因为当前代码不是统一 FPN。当前 [neck.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/neck.py) 明确按 `resnet18 / dla34 / mobilenetv3_large` 分支构建不同 decoder。

### Q2：为什么默认配置是 `resnet18`，但部署又推荐 `mobilenetv3_large`

这是当前代码状态决定的：

- 默认训练配置是 `resnet18`
- `mobilenetv3_large` 一般更轻，更适合端侧部署
- 如果目标是训练速度、模型轻量化和导出便利性，`mobilenetv3_large` 往往更省心

### Q3：如何新增 backbone

1. 在 [backbones](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/backbones) 下新增实现
2. 用 `@register_backbone("name")` 注册
3. 提供 `out_channels`
4. `forward()` 返回 4 层特征
5. 在 [neck.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/neck.py) 的 `build_decoder()` 中补对应 decoder

## 13. 推荐使用顺序

```bash
pip install -e .

python scripts/train.py --config configs/spot_centernet.yaml

python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input photos \
  --output outputs/infer_results

python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --output outputs/best.onnx
```

如果要直接面向 RKNN 部署，建议先把 `model.backbone` 切到 `mobilenetv3_large` 再训练和导出。
