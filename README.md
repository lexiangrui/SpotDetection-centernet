# 光斑质心定位系统 —— 基于 CenterNet 的点目标检测

## 1. 项目简介

本项目实现了一套基于 CenterNet 的单类点目标检测系统，用于从图像或视频中定位光斑中心点坐标。模型通过预测：

- 1 个通道的热力图 `heatmap`
- 2 个通道的亚像素偏移 `reg`

再经过局部极大值抑制和 Top-K 解码，输出每个光斑的 `(x, y, score)`。

支持内置3种骨干网络与 timm 扩展骨干：

- `dla34` — DLA-34
- `resnet18` — ResNet-18
- `mobilenetv3_large` — MobileNetV3-Large
- `timm:<model_name>` — 支持 `features_only=True` 的 timm 库里的模型

支持 4 种热力图损失函数：

- `focal` — Focal Loss（默认）
- `mse` — 前景 MSE Loss
- `bce` — 标准 BCE Loss（全图像素）
- `kl` — KL 散度 Loss

### 核心特性

- 单类点目标检测，类别名固定为 `spot`
- 多骨干网络切换：DLA-34 / ResNet-18 / MobileNetV3-Large / timm 库里的模型
- FPN 风格多尺度特征融合
- CenterNet 风格高斯热力图监督
- 自动适配 `CUDA / MPS / CPU`
- 图片和视频统一推理入口
- 三栏可视化输出：原图 | 热力图 | 检测结果
- ONNX / RKNN 部署导出

### 检测流程

```text
输入图像
  -> resize + pad 到固定输入尺寸
  -> Backbone (DLA-34 / ResNet-18 / MobileNetV3-Large / timm 库里的模型)
  -> FPNFusion (多尺度特征融合)
  -> CenterNet Head
       |- heatmap: [B, 1, H/4, W/4]
       |- reg:     [B, 2, H/4, W/4]
  -> sigmoid + local NMS + Top-K
  -> 映射回原图坐标
  -> 检测结果 (x, y, score)
```

---

## 2. 项目结构

```text
光斑定位-centernet/
├── pyproject.toml                       # Python 包配置
├── README.md
├── configs/
│   └── spot_centernet.yaml              # 模型/训练/推理配置
├── scripts/
│   ├── train.py                         # 训练脚本
│   ├── infer.py                         # 推理脚本（图片/视频）
│   ├── export_onnx.py                   # ONNX 导出
│   ├── export_rknn.py                   # RKNN 导出
│   └── make_rknn_dataset.py             # RKNN 量化数据集制备
├── src/centernet_spot/
│   ├── __init__.py
│   ├── backbones/                       # 骨干网络
│   │   ├── __init__.py                  #   注册表 + 自动加载
│   │   ├── registry.py                  #   backbone 注册/构建机制
│   │   ├── utils.py                     #   torchvision 权重解析
│   │   ├── timm_backbone.py             #   timm
│   │   ├── dla34.py                     #   DLA-34
│   │   ├── resnet18.py                  #   ResNet-18
│   │   └── mobilenetv3.py               #   MobileNetV3-Large
│   ├── neck.py                          # FPN 特征融合
│   ├── head.py                          # CenterNet 检测头
│   ├── model.py                         # SpotCenterNet 模型组装
│   ├── config.py                        # YAML 配置加载
│   ├── data.py                          # 数据集、高斯热力图生成
│   ├── decode.py                        # NMS + Top-K 解码
│   ├── losses.py                        # 损失函数（focal/mse/bce/kl + reg_l1）
│   ├── preprocessing.py                 # 统一预处理（推理/RKNN 共用）
│   ├── transforms.py                    # resize-pad、坐标正逆变换
│   ├── split.py                         # 训练/验证集划分
│   ├── utils.py                         # 种子、设备、归一化、IO 工具
│   └── visualization.py                 # 可视化（热力图、检测标注、损失曲线）
├── photos/                              # 图片数据
├── labels_raw/                          # Labelme 标注文件
├── splits/                              # 训练/验证划分文件
├── models/                              # 训练输出的模型权重
└── outputs/                             # 推理/导出输出
```

---

## 3. 环境配置

### 3.1 安装

推荐 Python `3.10+`。

```bash
# 克隆项目
cd 光斑定位-centernet

# 安装为可编辑包（推荐）
pip install -e .
```

这会自动安装所有依赖（torch、torchvision、timm、opencv-python、numpy、pyyaml、matplotlib）。

安装完成后，所有脚本可直接运行，无需额外配置 `PYTHONPATH`。

### 3.2 可选依赖

```bash
pip install labelme          # 制作点标注
pip install rknn-toolkit2    # RKNN 导出（仅 ARM 平台）
```

### 3.3 验证环境

```bash
python -c "import torch, timm; print(torch.__version__); print('timm=', timm.__version__); print('cuda=', torch.cuda.is_available()); print('mps=', torch.backends.mps.is_available())"
python -c "from centernet_spot import SpotCenterNet; print('import OK')"
```

---

## 4. 数据格式

### 4.1 图片目录

将图片放在 `photos/` 中，支持 `.jpg`、`.jpeg`、`.png`、`.bmp`。推荐按编号命名：

```text
photos/
├── 000001.jpg
├── 000002.jpg
└── ...
```

### 4.2 Labelme 标注

使用 Labelme 标注：

```bash
labelme photos/ --output labels_raw/
```

每个目标点使用：

- `label = "spot"`，`shape_type = "point"`

每张图必须额外画一条代表性尺寸线段：

- `label = "spot_size"`，`shape_type = "line"`

这条线段决定该图监督高斯核的直径。缺失会在数据集加载时报错。

示例标注 JSON：

```json
{
  "shapes": [
    {
      "label": "spot",
      "points": [[174.33, 126.66]],
      "shape_type": "point"
    },
    {
      "label": "spot_size",
      "points": [[300.0, 120.0], [316.0, 120.0]],
      "shape_type": "line"
    }
  ],
  "imagePath": "../photos/000008.jpg",
  "imageHeight": 401,
  "imageWidth": 644
}
```

### 4.3 训练/验证划分

`train.py` 会在每次训练前自动扫描 `labels_raw/`，按 `val_ratio` 重新划分并写入 `splits/train.txt` 和 `splits/val.txt`。

---

## 5. 配置文件

配置文件：`configs/spot_centernet.yaml`

### 5.1 当前默认配置值

| 字段 | 当前值 | 说明 |
| --- | --- | --- |
| `model.backbone` | `mobilenetv3_large` | 骨干网络 |
| `model.neck_channels` | `96` | FPN 通道数 |
| `model.head_channels` | `48` | 检测头通道数 |
| `model.backbone_kwargs.pretrained` | `true` | 使用预训练权重 |
| `model.input_normalization.mean` | `[0.485, 0.456, 0.406]` | ImageNet 均值 |
| `model.input_normalization.std` | `[0.229, 0.224, 0.225]` | ImageNet 标准差 |
| `data.input_width` | `640` | 输入宽度 |
| `data.input_height` | `640` | 输入高度 |
| `data.down_ratio` | `4` | 下采样率 |
| `data.max_objects` | `512` | 最大目标数 |
| `train.batch_size` | `4` | 批次大小 |
| `train.epochs` | `60` | 训练轮数 |
| `train.lr` | `0.0005` | 学习率 |
| `train.heatmap_loss_type` | `focal` | 热力图损失函数 |
| `infer.score_threshold` | `0.6` | 推理置信度阈值 |
| `infer.topk` | `256` | Top-K 候选数 |
| `infer.nms_kernel` | `5` | NMS 核大小 |

### 5.2 输入归一化

当 `model.input_normalization` 存在时，使用配置中指定的 mean/std。

当 `model.input_normalization` 不存在时：

- 内置骨干 `mobilenetv3_large` / `resnet18` / `dla34` 且 `pretrained=true`：回退到 ImageNet `mean=[0.485, 0.456, 0.406]`、`std=[0.229, 0.224, 0.225]`
- `timm:<model_name>` 且 `pretrained=true`：读取 `timm` 的 `pretrained_cfg` / `resolve_model_data_config(...)` 结果作为默认 mean/std
- 其他情况：使用恒等归一化，即 `mean=[0,0,0]`、`std=[1,1,1]`

实际执行的图像预处理流程固定为：

1. `BGR -> RGB`
2. 按目标输入尺寸做 `resize + pad`（保持长宽比，`cv2.INTER_LINEAR`，pad 值为 `0`）
3. 转 `float32` 并除以 `255.0`
4. 按 `(x / 255.0 - mean) / std` 做逐通道归一化

不直接照搬 `timm.data.create_transform(...)` 的分类推理流程；只复用 `timm` 默认数据配置中的归一化参数，不复用其 center crop / resize policy

`train.py`、`infer.py`、`make_rknn_dataset.py` 在启动时都会打印一条预处理摘要；如果走的是 `timm` 默认配置，还会额外打印 `input_size`、`interpolation`、`crop_pct` 等解析出的 `timm` 数据配置。

### 5.3 输入输出分辨率关系

默认 `640 × 640` 输入，`down_ratio = 4`，输出分辨率 `160 × 160`。

### 5.4 使用 timm backbone

> 具体文档可以看这里:
> timm.create_model / pretrained_cfg：https://huggingface.co/docs/timm/v1.0.25/en/reference/models
> create_transform / resolve_data_config 官方推理示例：https://huggingface.co/docs/timm/main/quickstart
> 特征提取：https://huggingface.co/docs/timm/feature_extraction

通过将 `model.backbone` 设置为 `timm:<model_name>` 来启用 timm

可运行 `tools\list_timm_backbones.py` 来查看所有可用的模型

示例：

```yaml
model:
  backbone: timm:convnext_tiny
  neck_channels: 128
  head_channels: 64
  backbone_kwargs:
    pretrained: true
    out_indices: [1, 2, 3, 4]
```

说明：

- 底层调用 `timm.create_model(..., features_only=True)`
- 仅支持能够返回多尺度特征的 timm 模型
- `backbone_kwargs` 会透传给 timm
- 常用参数包括：`pretrained`、`out_indices`、`in_chans`
- 若传入 `pretrained_cfg` 或 `pretrained_cfg_overlay`，默认归一化解析也会跟随该配置一起更新

如果某个 timm 模型不支持 `features_only=True`，或不能稳定返回当前 FPN 所需的多尺度特征，构建时会直接报错

兼容性说明：

- 当前适配方式与 `timm` 官方特征提取接口一致，核心依赖是 `features_only=True` 和 `feature_info`
- 这更适合 CNN、ConvNeXt、部分分层 Transformer
- 对纯分类风格、单尺度输出、或缺少合适 `feature_info` 的模型，不保证可直接接入

---

## 6. 训练

### 6.1 基本命令

```bash
python scripts/train.py --config configs/spot_centernet.yaml
```

### 6.2 可选参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径 |
| `--epochs` | 覆盖训练轮数 |
| `--batch-size` | 覆盖批次大小 |
| `--save-dir` | 覆盖输出目录 |

示例：

```bash
python scripts/train.py \
  --config configs/spot_centernet.yaml \
  --epochs 200 \
  --batch-size 8 \
  --save-dir models/my_experiment
```

使用 timm backbone 训练示例：

```bash
python scripts/train.py --config configs/spot_centernet.yaml
```

只需在 YAML 中把 `model.backbone` 改成类似 `timm:convnext_tiny`、`timm:efficientnet_b0`、`timm:resnet34` 即可。

训练开始时会先打印 `train_preprocessing`，用于确认当前 backbone 实际采用的输入预处理逻辑。

### 6.3 训练输出

训练目录中会生成：

| 文件 | 说明 |
| --- | --- |
| `best.pt` | 最优验证损失对应的模型权重 |
| `last.pt` | 最后一轮的模型权重 |
| `metrics.json` | 每轮训练/验证指标记录 |
| `loss_curve.png` | 训练/验证损失曲线图 |
| `train_vis/epoch_XXX.jpg` | 每 20 轮保存一次可视化对比 |

### 6.4 切换损失函数

在配置文件中修改 `train.heatmap_loss_type`：

```yaml
train:
  heatmap_loss_type: focal   # 可选: focal, mse, bce, kl
```

---

## 7. 推理

### 7.1 基本命令

#### 单张图片

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --input photos/000008.jpg \
  --output outputs/infer_single
```

#### 整个图片目录

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --input photos \
  --output outputs/infer_all
```

#### 视频

```bash
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --input video/capture.mp4 \
  --output outputs/infer_video
```

### 7.2 参数说明

| 参数 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config` | 否 | `configs/spot_centernet.yaml` | 配置文件 |
| `--checkpoint` | 是 | — | 模型权重路径 |
| `--input` | 是 | — | 图片、视频或目录 |
| `--output` | 否 | `outputs/infer` | 输出目录 |
| `--score-threshold` | 否 | 配置文件中的值 | 置信度阈值 |
| `--topk` | 否 | 配置文件中的值 | 候选点数量上限 |

脚本启动时会打印 `infer_preprocessing`，用于确认当前 checkpoint 和配置组合下的实际预处理逻辑

### 7.3 输出结果

#### 图片输出

每张图片生成 `{name}_vis.jpg`（三栏可视化）和 `{name}.json`：

```json
{
  "type": "image",
  "image": "photos/000008.jpg",
  "count": 25,
  "detections": [
    { "score": 0.9767, "x": 124.52, "y": 223.27, "class_id": 0 }
  ]
}
```

#### 视频输出

每个视频生成 `{name}_vis.mp4`（或 `.avi`）和 `{name}.json`：

```json
{
  "type": "video",
  "video": "video/capture.mp4",
  "fps": 25.0,
  "frame_count": 120,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0.0,
      "count": 18,
      "detections": [
        { "score": 0.98, "x": 261.22, "y": 203.26, "class_id": 0 }
      ]
    }
  ]
}
```

---

## 8. ONNX 与 RKNN 导出

### 8.1 导出 ONNX

```bash
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --output outputs/best.onnx
```

| 参数 | 说明 |
| --- | --- |
| `--checkpoint` | 输入 `.pt` 权重 |
| `--config` | 可选；若 checkpoint 内含配置则自动使用 |
| `--output` | 输出 `.onnx` 路径 |
| `--opset` | ONNX opset 版本，默认 `17` |
| `--batch-size` | 导出 batch size，默认 `1` |
| `--dynamic-batch` | 启用动态 batch |

> 注意：部分 timm backbone 可能包含对 ONNX 不友好的算子，是否能成功导出取决于具体模型

### 8.2 生成 RKNN 量化数据集

```bash
python scripts/make_rknn_dataset.py \
  --config configs/spot_centernet.yaml \
  --photos photos \
  --out-dir outputs/rknn_dataset \
  --dataset-txt outputs/rknn_dataset.txt \
  --limit 32
```
启动时会打印 `rknn_dataset_preprocessing`，便于确认量化数据集和训练/推理使用的是同一套预处理

### 8.3 ONNX 转 RKNN

```bash
python scripts/export_rknn.py \
  --onnx outputs/best.onnx \
  --output outputs/best.rknn \
  --target-platform rk3576
```

启用量化：

```bash
python scripts/export_rknn.py \
  --onnx outputs/best.onnx \
  --output outputs/best_int8.rknn \
  --target-platform rk3576 \
  --quantize \
  --dataset outputs/rknn_dataset.txt
```

> ps:并非所有 timm backbone 都适合 RKNN 部署

---

## 9. 调参建议

| 现象 | 建议 |
| --- | --- |
| 漏检较多 | 降低 `infer.score_threshold`；检查 `spot_size` 标定是否偏小 |
| 误检较多 | 提高 `infer.score_threshold`；增加训练样本 |
| 相邻光斑粘连 | 减小 `infer.nms_kernel`；检查 `spot_size` 标定线是否过长 |
| 训练不稳定 | 降低 `train.lr`；增大 `train.batch_size`；减小 `neck_channels` |
| 显存不足 | 降低 `batch_size` → 降低输入分辨率 → 使用 `resnet18` / 轻量 timm 模型 → 减小通道数 |

---

## 10. 核心源码说明

| 文件 | 说明 |
| --- | --- |
| `backbones/` | 骨干网络注册表与各实现（DLA-34、ResNet-18、MobileNetV3、timm 适配器） |
| `neck.py` | FPN 多尺度特征融合 |
| `head.py` | CenterNet 检测头（heatmap + reg） |
| `model.py` | SpotCenterNet 模型组装（backbone → neck → head） |
| `data.py` | Labelme 标注读取、高斯热力图生成、数据增强 |
| `losses.py` | 热力图损失（focal/mse/bce/kl）+ 回归 L1 损失 |
| `decode.py` | 局部 NMS → Top-K → 偏移量修正 → 映射回原图坐标 |
| `preprocessing.py` | 统一预处理（推理/RKNN 数据集共用） |
| `transforms.py` | resize-pad 变换、坐标正逆映射 |
| `visualization.py` | 热力图渲染、检测标注绘制、损失曲线绘制 |
| `split.py` | 训练/验证集划分与读写 |
| `utils.py` | 种子设定、设备选择、输入归一化、JSON/目录工具 |
| `config.py` | YAML 配置加载 |

---

## 11. 常见问题

### Q1：训练时报 `No samples found for split=train`

- `labels_raw/` 为空或标注文件中没有 `label=spot` 且 `shape_type=point` 的标注
- 确认标注文件数量足够（至少 2 个以上才能划分训练和验证集）

### Q2：推理时没有输出结果

- 检查 `--checkpoint` 路径是否正确
- 检查 `--input` 是否为支持的图片/视频文件或目录
- 尝试降低 `--score-threshold`

### Q3：如何新增骨干网络

两种方式：

1. **直接使用 timm**：

```yaml
model:
  backbone: timm:convnext_tiny
  backbone_kwargs:
    pretrained: true
    out_indices: [1, 2, 3, 4]
```

2. **手写注册一个自定义 backbone**：

   1. 在 `src/centernet_spot/backbones/` 下创建新文件
   2. 用 `@register_backbone("name")` 装饰器注册
   3. 类需提供 `out_channels: List[int]` 属性
   4. `forward()` 返回多尺度特征列表 `List[torch.Tensor]`
   5. 在 `backbones/__init__.py` 中导入新模块

然后在配置文件中设置：

```yaml
model:
  backbone: name
```

### Q4：为什么某些 timm 模型不能直接用

因为当前适配方式依赖：

```python
timm.create_model(model_name, features_only=True)
```

所以要求该模型：

- 支持 `features_only=True`
- 能输出多尺度特征
- 输出层数和通道数适合接入当前 FPN

如果不满足这些条件，需要单独写适配器

---

## 12. 推荐使用顺序

```bash
# 0) 安装
pip install -e .

# 1) 训练
python scripts/train.py --config configs/spot_centernet.yaml

# 2) 推理
python scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --input photos \
  --output outputs/infer_results

# 3) 导出 ONNX
python scripts/export_onnx.py \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --output outputs/best.onnx

# 4) 如需 RKNN，先准备量化数据，再转换
python scripts/make_rknn_dataset.py \
  --photos photos \
  --out-dir outputs/rknn_dataset \
  --dataset-txt outputs/rknn_dataset.txt

python scripts/export_rknn.py \
  --onnx outputs/best.onnx \
  --output outputs/best.rknn \
  --target-platform rk3576
```
