# 光斑质心定位系统

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
│   │   └── resnet18.py
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

- 当前项目固定使用 ImageNet mean/std

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
- `best_loss.pt`
- `train.log`
- `metrics.jsonl`
- `metrics.csv`
- `metrics.json`
- `summary.json`
- `run_context.json`
- `loss_curve.png`
- `tensorboard/events.out.tfevents.*`
- `train_vis/epoch_XXX.jpg`

当前训练监控方式：

- 终端使用 `tqdm` 实时显示训练、验证、评估进度与吞吐。
- `train.log` 保存完整文本日志，适合长期跑实验时回看。
- `metrics.jsonl` / `metrics.csv` 保存逐轮结构化指标，方便 pandas、Excel 或自定义脚本分析。
- TensorBoard 记录 `train/val loss`、`AP`、`F1`、`fitness`、`precision`、`recall`、学习率和样本可视化。

启动 TensorBoard：

```bash
tensorboard --logdir models/spot_centernet_resnet18_focal/tensorboard
```

当前训练控制策略：

- `ReduceLROnPlateau` 监控 `val_loss`，比直接盯 `F1` 更平滑，降学习率更稳定。
- `best.pt` 与 early stopping 依据 `fitness = 0.7 * AP + 0.3 * F1` 选优。
- `AP` 用整条 PR 曲线衡量排序质量，避免只盯单阈值 `F1` 带来的波动。

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

当前项目已经固定为 `resnet18 backbone + 3-stage deconv decoder`，不再保留其他 backbone 分支和对应 decoder。

## 10. 调参建议

| 现象 | 建议 |
| --- | --- |
| 漏检较多 | 降低 `infer.score_threshold`；检查 `spot_size` 标定是否偏小 |
| 误检较多 | 提高 `infer.score_threshold`；增加训练样本 |
| 相邻光斑粘连 | 减小 `infer.nms_kernel`；检查 `spot_size` 标定线是否过长 |
| 训练不稳定 | 降低 `train.lr`；增大 `train.batch_size` |
| 部署导出失败 | 先检查当前 checkpoint 和配置文件是否匹配，再检查 ONNX/RKNN 环境版本 |

## 11. 核心源码说明

| 文件 | 说明 |
| --- | --- |
| `backbones/resnet18.py` | 固定使用的 ResNet-18 backbone |
| `neck.py` | ResNet-18 对应 decoder 实现 |
| `head.py` | heatmap/reg 检测头 |
| `model.py` | backbone + decoder + head 组装 |
| `data.py` | 数据加载与监督目标生成 |
| `decode.py` | NMS、Top-K、坐标反变换 |
| `preprocessing.py` | 推理与 RKNN 共享预处理 |
| `transforms.py` | letterbox 与坐标映射 |
| `visualization.py` | 可视化与训练曲线导出 |
| `utils.py` | 设备、归一化、JSON/目录工具 |

## 12. 常见问题

### Q1：为什么配置文件里没有 `model.backbone` 这些参数了

因为当前项目已经固定为 `resnet18`，对应的 decoder 和归一化也一起写死了，不再保留多 backbone 选择。

### Q2：为什么导出/推理时不再加载预训练 backbone

因为导出和推理都会先加载完整 checkpoint，backbone 预训练初始化在这两个阶段是多余的，还可能触发不必要的权重下载。

### Q3：如果后面还要换 backbone 怎么办

当前代码已经刻意去掉了通用注册表和分支逻辑；如果后面要换 backbone，需要直接改 [backbones/resnet18.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/backbones/resnet18.py)、[neck.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/neck.py) 和 [model.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/model.py)。

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
