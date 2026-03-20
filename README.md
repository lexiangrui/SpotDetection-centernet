# 光斑质心定位系统 —— 基于 CenterNet 的点目标检测

## 1. 项目简介

本项目实现了一套基于 CenterNet 的单类点目标检测系统，用于从图像或视频中定位光斑中心点坐标。模型通过预测：

- 1 个通道的热力图 `heatmap`
- 2 个通道的亚像素偏移 `reg`

再经过局部极大值抑制和 Top-K 解码，输出每个光斑的 `(x, y, score)`。

当前代码支持 3 种骨干网络：

- `dla34`
- `resnet18`
- `unet`

当前仓库还包含完整的训练、图片/视频推理、ONNX 导出、RKNN 量化数据准备和 RKNN 导出脚本。

### 核心特性

- 单类点目标检测，类别名固定为 `spot`
- 多骨干网络切换：DLA-34 / ResNet-18 / U-Net
- FPN 风格多尺度特征融合
- CenterNet 风格高斯热力图监督
- 自动适配 `CUDA / MPS / CPU`
- 图片和视频统一推理入口
- 三栏可视化输出：原图 | 热力图 | 检测结果

### 当前实现流程

```text
输入图像
  -> resize + pad 到固定输入尺寸
  -> Backbone (DLA-34 / ResNet-18 / U-Net)
  -> FPNFusion
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
├── README.md
├── README.html
├── configs/
│   ├── spot_centernet_dla34.yaml
│   ├── spot_centernet_resnet18.yaml
│   └── spot_centernet_unet.yaml
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── export_onnx.py
│   ├── onnx2rknn.py
│   ├── export_rknn.py
│   └── make_rknn_dataset.py
├── src/centernet_spot/
│   ├── config.py
│   ├── data.py
│   ├── decode.py
│   ├── losses.py
│   ├── model.py
│   ├── split.py
│   ├── transforms.py
│   └── utils.py
├── photos/
├── video/
├── labels_raw/
├── splits/
└── outputs/
```

### 当前仓库示例数据状态

截至当前仓库内容：

- `photos/` 中有 26 张图片
- `labels_raw/` 中有 4 份标注
- `video/` 中有 1 个视频
- `splits/train.txt` 当前为 3 个样本
- `splits/val.txt` 当前为 1 个样本

这些数字只是当前仓库内容，不是代码限制。

---

## 3. 环境配置

### 3.1 Python 版本

推荐 Python `3.10+`。

### 3.2 基础依赖

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pyyaml
```

可选依赖：

- `matplotlib`：训练后绘制 `loss_curve.png`
- `labelme`：制作点标注
- `rknn-toolkit2`：RKNN 导出

```bash
pip install matplotlib labelme
```

### 3.3 验证 PyTorch 环境

```bash
python3 -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print('mps=', torch.backends.mps.is_available())"
```

---

## 4. 数据格式

### 4.1 图片目录

将图片放在 `photos/` 中，支持：

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

推荐按编号命名，例如：

```text
photos/
├── 000008.jpg
├── 000009.jpg
├── 000015.jpg
└── 000056.jpg
```

### 4.2 Labelme 标注格式


```bash
   labelme photos/ --output labels_raw/
   ```

每个目标点使用：

- `label = "spot"`
- `shape_type = "point"`

每张图必须额外画一条代表性尺寸线段：

- `label = "spot_size"`
- `shape_type = "line"`

这条线段直接决定该图监督高斯核的直径。当前代码会先把这条线段映射到输出热力图坐标系，再用映射后的直径绘制高斯核；如果缺失，数据集加载会直接报错。

示例：

```json
{
  "version": "5.11.2",
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

当前 `train.py` 会在每次训练开始前自动：

- 扫描 `labels_raw/` 下的全部标注
- 按配置文件中的 `data.val_ratio` 和 `seed` 重新划分训练集 / 验证集
- 覆盖写入 `splits/train.txt` 和 `splits/val.txt`

训练自动生成后的输出是：

- `splits/train.txt`
- `splits/val.txt`

---

## 5. 配置文件


- `configs/spot_centernet.yaml`



### 5.1 当前默认配置值

以下默认值对应当前仓库中的配置文件：

| 字段 | 当前值 |
| --- | --- |
| `data.input_width` | `512` |
| `data.input_height` | `512` |
| `data.down_ratio` | `4` |
| `data.max_objects` | `512` |
| `train.batch_size` | `4` |
| `train.epochs` | `120` |
| `train.lr` | `0.001` |
| `infer.score_threshold` | `0.3` |
| `infer.topk` | `256` |
| `infer.nms_kernel` | `9` |

### 5.2 关键参数说明

当前代码不再使用 `normalize_mean` 和 `normalize_std`。输入图像预处理统一为：

- `BGR -> RGB`
- `image / 255.0`

因此模型输入范围现在是 `[0, 1]`。

### 5.3 输入输出分辨率关系

当前默认输入尺寸为 `512 x 512`，`down_ratio = 4`，因此模型输出分辨率为：

```text
512 x 512 -> 128 x 128
```

---

## 6. 训练

### 6.1 基本命令



```bash
python3 scripts/train.py --config configs/spot_centernet.yaml

```

### 6.2 可选参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径 |
| `--epochs` | 覆盖训练轮数 |
| `--batch-size` | 覆盖 batch size |
| `--save-dir` | 覆盖输出目录 |

示例：

```bash
python3 scripts/train.py \
  --config configs/spot_centernet_resnet18.yaml \
  --epochs 200 \
  --batch-size 8 \
  --save-dir outputs/spot_centernet_resnet18_exp
```

### 6.3 训练输出

训练目录中会生成：

- `best.pt`
- `last.pt`
- `metrics.json`
- `loss_curve.png`
- `train_vis/epoch_XXX.jpg`（每 20 个 epoch 保存一次可视化）

### 6.4 当前训练流程

1. 从 `labels_raw/` 读取 Labelme 标注
2. 自动重建 `train.txt / val.txt`
3. 将图像 `resize + pad` 到固定输入尺寸
4. 生成热力图监督和偏移量监督
5. 前向输出 `heatmap` 和 `reg`
6. 使用 `focal_loss + reg_l1_loss`
7. 每轮保存 `last.pt`
8. 当验证损失更低时更新 `best.pt`

---

## 7. 推理

### 7.1 基本命令

`infer.py` 当前代码里的 `--config` 默认值同样是不存在的 `configs/spot_centernet.yaml`，因此推理时也请显式传入配置文件。

#### 单张图片

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18/best.pt \
  --input photos/000008.jpg \
  --output outputs/infer_single
```

#### 整个图片目录

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18_focal/best.pt \
  --input photos \
  --output outputs/spot_centernet_resnet18_focal
```


#### 单个视频

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_mobilenetv3_focal/best.pt \
  --input video/capture.mp4 \
  --output outputs/infer_video
```

#### 整个视频目录

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_resnet18/best.pt \
  --input video \
  --output outputs/infer_video
```
```bash
python3 scripts/infer.py \
  --config configs/spot_centernet.yaml \
  --checkpoint models/spot_centernet_dla34/best.pt \
  --input video \
  --output outputs/infer_video
```
### 7.2 参数说明

| 参数 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config` | 建议显式传入 | `configs/spot_centernet.yaml` | 当前默认值在仓库中不存在 |
| `--checkpoint` | 是 | 无 | 模型权重 |
| `--input` | 是 | 无 | 图片、视频或目录 |
| `--output` | 否 | `outputs/infer` | 输出目录 |
| `--score-threshold` | 否 | 配置文件中的值 | 置信度阈值 |
| `--topk` | 否 | 配置文件中的值 | 候选点数量上限 |

### 7.3 输出结果

#### 图片输入

每张图片会生成：

- `{name}_vis.jpg`
- `{name}.json`

JSON 示例：

```json
{
  "type": "image",
  "image": "photos/000008.jpg",
  "count": 25,
  "detections": [
    {
      "score": 0.9767,
      "x": 124.52,
      "y": 223.27,
      "class_id": 0
    }
  ]
}
```

#### 视频输入

每个视频会生成：

- `{name}_vis.mp4`
- 若 MP4 编码不可用则回退为 `{name}_vis.avi`
- `{name}.json`

视频 JSON 示例：

```json
{
  "type": "video",
  "video": "video/capture.mp4",
  "visualization_video": "outputs/infer_video/capture_vis.mp4",
  "fps": 25.0,
  "frame_count": 120,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0.0,
      "count": 18,
      "detections": [
        {
          "score": 0.98,
          "x": 261.22,
          "y": 203.26,
          "class_id": 0
        }
      ]
    }
  ]
}
```

---

## 8. ONNX 与 RKNN 导出

### 8.1 导出 ONNX

`scripts/export_onnx.py` 支持：

- 从 `--config` 加载配置
- 或直接使用 checkpoint 内部保存的 `config`

示例：

```bash
python3 scripts/export_onnx.py \
  --checkpoint models/spot_centernet_resnet18/best.pt \
  --output outputs/spot_centernet_resnet18/best.onnx
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--checkpoint` | 输入 `.pt` 权重 |
| `--config` | 可选；若 checkpoint 中没有配置则必须提供 |
| `--output` | 输出 `.onnx` 路径 |
| `--opset` | ONNX opset，默认 `17` |
| `--batch-size` | 导出时的 batch size，默认 `1` |
| `--dynamic-batch` | 导出动态 batch 维度 |

导出输出为两个张量：

- `heatmap`
- `reg`

### 8.2 生成 RKNN 量化数据集

`scripts/make_rknn_dataset.py` 会把图片预处理后保存为 `.npy`，再生成一个数据列表文本。

注意：

- 这个脚本当前带有面向本地环境的默认绝对路径
- 实际使用时建议始终显式传参

示例：

```bash
python3 scripts/make_rknn_dataset.py \
  --photos photos \
  --out-dir outputs/spot_centernet_resnet18/rknn_dataset \
  --dataset-txt outputs/spot_centernet_resnet18/rknn_dataset.txt \
  --limit 32 \
  --input-width 512 \
  --input-height 512
```

### 8.3 ONNX 转 RKNN

推荐使用 `scripts/onnx2rknn.py`。当前模型输入只做 `image / 255.0`，因此该脚本会为 RKNN 固定写入：

- `mean_values = [[0, 0, 0]]`
- `std_values = [[255, 255, 255]]`

```bash
python3 scripts/onnx2rknn.py \
  --onnx outputs/spot_centernet_resnet18/best.onnx \
  --output outputs/spot_centernet_resnet18/best.rknn \
  --config configs/spot_centernet_resnet18.yaml \
  --target-platform rk3576
```

启用量化：

```bash
python3 scripts/onnx2rknn.py \
  --onnx outputs/spot_centernet_resnet18/best.onnx \
  --output outputs/spot_centernet_resnet18/best_int8.rknn \
  --config configs/spot_centernet_resnet18.yaml \
  --target-platform rk3576 \
  --quantize \
  --dataset outputs/spot_centernet_resnet18/rknn_dataset.txt
```

### 8.4 `export_rknn.py` 说明

仓库中还保留了 `scripts/export_rknn.py`。它现在也会读取配置文件并传入 RKNN 预处理参数，但它仍带有本地机器风格的默认 `--onnx` / `--output` 路径，因此建议：

- 不依赖默认值
- 始终显式传入 `--onnx --output --config`

---

## 9. 调参建议

### 9.1 漏检较多

1. 降低 `infer.score_threshold`
2. 检查 `spot_size` 标定是否偏小

### 9.2 误检较多

1. 提高 `infer.score_threshold`
2. 检查 `spot_size` 标定是否偏大
3. 增加训练样本

### 9.3 相邻光斑容易粘连

1. 减小 `infer.nms_kernel`
2. 检查 `spot_size` 标定线是否过长

### 9.4 训练不稳定

1. 降低 `train.lr`
2. 增大 `train.batch_size`
3. 减小 `model.neck_channels`
4. 先使用 `resnet18` 排查数据问题

---

## 10. 核心源码说明

| 文件 | 说明 |
| --- | --- |
| `src/centernet_spot/config.py` | 加载 YAML 配置 |
| `src/centernet_spot/data.py` | 读取 Labelme、预处理、生成 heatmap/reg 监督 |
| `src/centernet_spot/model.py` | Backbone、FPNFusion、CenterNet Head、总模型封装 |
| `src/centernet_spot/losses.py` | `focal_loss` 和 `reg_l1_loss` |
| `src/centernet_spot/decode.py` | 局部 NMS、Top-K、偏移量修正、回原图坐标 |
| `src/centernet_spot/transforms.py` | `resize_and_pad_image`、坐标映射、逆映射工具 |
| `src/centernet_spot/split.py` | 划分训练/验证集并读写 split 文件 |
| `src/centernet_spot/utils.py` | 随机种子、设备选择、目录创建、JSON 保存 |

---

## 11. 常见问题

### Q1：训练时报 `No samples found for split=train`

通常是以下原因之一：

- `labels_raw/` 为空
- `splits/train.txt` 不存在且标注文件数量太少
- 标注文件存在，但点标注不是 `label=spot` 且 `shape_type=point`

建议先执行：

训练脚本会自动刷新划分文件。

### Q2：推理时报找不到配置文件

当前 `train.py` 和 `infer.py` 代码中的默认 `--config` 都是：

```text
configs/spot_centernet.yaml
```

但这个文件当前不在仓库里，所以请显式传：

```bash
--config configs/spot_centernet_resnet18.yaml
```

或另外两个现有配置文件。

### Q3：推理时没有输出结果

检查：

- `--checkpoint` 是否正确
- `--input` 是否为图片、视频或包含这些文件的目录
- `infer.score_threshold` 是否过高

### Q4：如何新增骨干网络

在 [src/centernet_spot/model.py](/Users/lexiangrui/Desktop/光斑定位-centernet/src/centernet_spot/model.py) 中：

1. 实现新的 `nn.Module`
2. 用 `@register_backbone("name")` 注册
3. 提供 `out_channels`
4. `forward()` 返回多尺度特征列表

然后在配置文件里设置：

```yaml
model:
  backbone: name
```

### Q5：显存不足怎么办

可按顺序尝试：

1. 降低 `train.batch_size`
2. 降低输入分辨率
3. 使用 `resnet18`
4. 减小 `neck_channels` 和 `head_channels`

---

## 12. 推荐使用顺序

```bash
# 1) 训练（会自动生成 train/val 划分）
python3 scripts/train.py --config configs/spot_centernet_resnet18.yaml

# 2) 推理
python3 scripts/infer.py \
  --config configs/spot_centernet_resnet18.yaml \
  --checkpoint models/spot_centernet_resnet18/best.pt \
  --input photos \
  --output outputs/infer_resnet18

# 3) 导出 ONNX
python3 scripts/export_onnx.py \
  --checkpoint models/spot_centernet_resnet18/best.pt \
  --output outputs/spot_centernet_resnet18/best.onnx

# 4) 如需 RKNN，先准备量化数据，再转换
python3 scripts/make_rknn_dataset.py \
  --photos photos \
  --out-dir outputs/spot_centernet_resnet18/rknn_dataset \
  --dataset-txt outputs/spot_centernet_resnet18/rknn_dataset.txt

python3 scripts/onnx2rknn.py \
  --onnx outputs/spot_centernet_resnet18/best.onnx \
  --output outputs/spot_centernet_resnet18/best.rknn \
  --config configs/spot_centernet_resnet18.yaml \
  --target-platform rk3576
```
