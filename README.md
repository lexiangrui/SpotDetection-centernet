# 光斑质心定位系统 —— 基于 CenterNet 的点目标检测

## 1. 项目简介

本项目实现了一套基于 **CenterNet** 的光斑质心定位系统，用于从图像中自动检测并定位光斑中心点坐标。系统采用 **Anchor-Free** 的检测范式，通过预测热力图（Heatmap）和亚像素偏移量（Offset）来定位每个光斑的质心位置，避免了传统目标检测中锚框设计的复杂性。

### 核心特性

- **单类点目标检测**：专注于 `spot`（光斑）类别的质心定位
- **双骨干网络支持**：提供 `DLA-34` 和 `ResNet-18` 两种骨干网络，可根据精度/速度需求灵活切换
- **FPN 多尺度融合**：通过特征金字塔网络融合多层特征，提升小目标检测能力
- **高斯热力图编码**：采用 CenterNet 风格的 Gaussian Heatmap 对点目标进行编码
- **Focal Loss 训练**：使用 Focal Loss 解决正负样本不平衡问题
- **三栏可视化输出**：推理时自动生成 原图 | 热力图 | 检测标注 的对比可视化图
- **自动设备适配**：自动检测 CUDA / MPS / CPU 并选择最优计算设备

### 技术架构

```
输入图像 → 仿射变换预处理 → Backbone(DLA-34/ResNet-18/U-Net) → FPN Neck → CenterNet Head
                                                                        ├── Heatmap Head (1 通道)
                                                                        └── Offset Head  (2 通道)
                                                          → Top-K 解码 + NMS → 检测结果（x, y, score）
```

---

## 2. 项目结构

```
光斑定位-centernet/
├── README.md                          # 本说明文档
├── configs/                           # 配置文件目录
│   ├── spot_centernet_dla34.yaml      # DLA-34 骨干网络配置
│   ├── spot_centernet_resnet18.yaml   # ResNet-18 骨干网络配置
│   └── spot_centernet_unet.yaml       # U-Net 骨干网络配置
├── scripts/                           # 可执行脚本
│   ├── make_splits.py                 # 数据集划分脚本
│   ├── train.py                       # 训练入口脚本
│   └── infer.py                       # 推理入口脚本
├── src/centernet_spot/                # 核心源码包
│   ├── __init__.py                    # 包初始化
│   ├── config.py                      # YAML 配置加载
│   ├── data.py                        # 数据集与高斯编码
│   ├── decode.py                      # 热力图解码（NMS + Top-K）
│   ├── losses.py                      # Focal Loss + Offset L1 Loss
│   ├── model.py                       # 模型定义（Backbone + Neck + Head）
│   ├── split.py                       # 训练/验证集划分逻辑
│   ├── transforms.py                  # 仿射变换工具
│   └── utils.py                       # 通用工具函数
├── photos/                            # 原始图像目录（共 56 张）
├── labels_raw/                        # Labelme 标注文件（已标注 8 张）
├── splits/                            # 数据划分文件
│   ├── train.txt                      # 训练集样本 ID 列表
│   └── val.txt                        # 验证集样本 ID 列表
└── outputs/                           # 输出目录
    ├── spot_centernet_dla34/          # DLA-34 训练输出
    │   ├── best.pt                    # 最优权重
    │   ├── last.pt                    # 最新权重
    │   └── metrics.json               # 训练指标记录
    ├── spot_centernet_resnet18/       # ResNet-18 训练输出
    │   └── metrics.json
    ├── spot_centernet_unet/           # U-Net 训练输出
    │   └── metrics.json
    ├── infer_dla34/                   # DLA-34 推理结果
    ├── infer_resnet18/                # ResNet-18 推理结果
    └── infer_unet/                    # U-Net 推理结果
```

---

## 3. 环境配置

### 3.1 Python 版本

推荐使用 **Python 3.10+**。

### 3.2 依赖安装

```bash
pip install torch torchvision   # PyTorch（根据 CUDA 版本选择对应安装命令）
pip install opencv-python       # OpenCV
pip install numpy               # NumPy
pip install pyyaml              # YAML 配置解析
```

> **GPU 加速说明**：
> - 如有 NVIDIA GPU，请安装对应 CUDA 版本的 PyTorch，参见 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
> - macOS Apple Silicon 用户可使用 MPS 加速，无需额外配置
> - 无 GPU 环境下自动回退到 CPU 运行

### 3.3 验证安装

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## 4. 数据准备

### 4.1 图像数据

将所有原始图像（支持 `.jpg`、`.jpeg`、`.png`、`.bmp` 格式）放置于 `photos/` 目录下，命名建议使用统一的编号格式，如：

```
photos/
├── 000001.jpg
├── 000002.jpg
├── ...
└── 000056.jpg
```

### 4.2 标注数据

本项目使用 [Labelme](https://github.com/labelmeai/labelme) 进行标注，光斑中心仍使用 **point（点标注）**，并且每张图额外标定一个代表性光斑尺寸。

#### 安装 Labelme

```bash
pip install labelme
```

#### 标注步骤

1. 启动 Labelme：
   ```bash
   labelme photos/ --output labels_raw/
   ```
2. 打开一张图片后，选择 **Create Point** 工具
3. 在每个光斑的中心位置点击标注一个点
4. 点标注的标签名称设置为 `spot`
5. 再选择 **Create Line** 工具，在一个代表性光斑上画一条直径线段
6. 这条尺寸标定线的标签名称设置为 `spot_size`
7. 保存后会在 `labels_raw/` 目录生成对应的 `.json` 文件

#### 标注文件格式

每个 `.json` 标注文件结构如下（Labelme 自动生成）：

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
      "label": "spot",
      "points": [[216.19, 125.50]],
      "shape_type": "point"
    },
    {
      "label": "spot_size",
      "points": [[300.00, 120.00], [316.00, 120.00]],
      "shape_type": "line"
    }
  ],
  "imagePath": "../photos/000001.jpg",
  "imageHeight": 401,
  "imageWidth": 644
}
```

> **注意**：
> - `label` 必须为 `spot`（与配置文件中的 `data.class_name` 一致）
> - 光斑中心点的 `shape_type` 必须为 `point`
> - 每张图建议额外标一条 `label=spot_size`、`shape_type=line` 的线段，长度表示该图的代表性光斑直径
> - 训练时会优先使用 `spot_size` 线段长度生成当前图片的 GT 高斯半径；如果缺失，则回退到配置里的固定 `point_box_width/point_box_height`
> - `imagePath` 是相对于标注文件的路径，Labelme 会自动生成

### 4.3 数据集划分

标注完成后，执行划分脚本将数据分为训练集和验证集：

```bash
python3 scripts/make_splits.py --root .
```

#### 可选参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--root` | `.` | 项目根目录路径 |
| `--label-dir` | `labels_raw` | 标注文件目录名 |
| `--split-dir` | `splits` | 划分文件输出目录名 |
| `--val-ratio` | `0.25` | 验证集比例（25%） |
| `--seed` | `42` | 随机种子，确保划分可复现 |

#### 输出结果

执行后会在 `splits/` 目录生成两个文件：

- `train.txt`：训练集样本 ID（每行一个，如 `000001`）
- `val.txt`：验证集样本 ID

```
labeled samples: 8
train samples: 6
val samples: 2
```

---

## 5. 模型训练

### 5.1 基本训练命令

使用 DLA-34 骨干网络训练：

```bash
python3 scripts/train.py --config configs/spot_centernet_dla34.yaml
```

使用 ResNet-18 骨干网络训练：

```bash
python3 scripts/train.py --config configs/spot_centernet_resnet18.yaml
```

使用 U-Net 骨干网络训练：

```bash
python3 scripts/train.py --config configs/spot_centernet_unet.yaml
```

### 5.2 命令行可选参数

训练脚本支持通过命令行覆盖配置文件中的部分参数：

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--config` | 配置文件路径 | `--config configs/spot_centernet_dla34.yaml` |
| `--epochs` | 覆盖训练轮数 | `--epochs 200` |
| `--batch-size` | 覆盖批大小 | `--batch-size 8` |
| `--save-dir` | 覆盖模型输出目录 | `--save-dir outputs/my_experiment` |

#### 完整示例

```bash
python3 scripts/train.py \
  --config configs/spot_centernet_dla34.yaml \
  --epochs 200 \
  --batch-size 8 \
  --save-dir outputs/dla34_exp2
```

### 5.3 训练过程输出

训练过程中会实时打印损失信息：

```json
{"step": 5, "loss": 2.840781, "hm_loss": 1.845416, "reg_loss": 0.995365}
```

每个 epoch 结束后会打印训练集和验证集的综合指标：

```json
{"epoch": 10, "train": {"loss": 0.523, "hm_loss": 0.312, "reg_loss": 0.211}, "val": {"loss": 0.618, "hm_loss": 0.405, "reg_loss": 0.213}}
```

### 5.4 训练输出文件

训练完成后，在输出目录（默认 `outputs/spot_centernet_dla34/`、`outputs/spot_centernet_resnet18/` 或 `outputs/spot_centernet_unet/`）中会生成：

| 文件 | 说明 |
| --- | --- |
| `best.pt` | 验证集损失最低时保存的最优模型权重 |
| `last.pt` | 最后一个 epoch 的模型权重 |
| `metrics.json` | 完整训练历史记录（每个 epoch 的 loss 数据） |

### 5.5 训练流程说明

1. **数据加载**：从 `labels_raw/` 读取 Labelme 标注，加载对应图像
2. **仿射变换**：将原始图像统一变换到 `640×384` 的网络输入尺寸
3. **数据增强**（仅训练集）：
   - 亮度随机扰动（±15%）
   - 高斯噪声注入（σ=0.01）
4. **热力图编码**：对每个标注点生成 Gaussian Heatmap，同时计算亚像素偏移量
5. **前向推理**：图像经过 Backbone → FPN Neck → Head，输出 Heatmap 和 Offset
6. **损失计算**：
   - **Heatmap Focal Loss**：监督热力图预测
   - **Offset L1 Loss**：监督偏移量回归
7. **模型保存**：每个 epoch 保存 `last.pt`，验证集 loss 更优时额外保存 `best.pt`

---

## 6. 模型推理

### 6.1 基本推理命令

#### 对整个目录推理

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet_dla34.yaml \
  --checkpoint outputs/spot_centernet_dla34/best.pt \
  --input photos \
  --output outputs/infer_dla34

python3 scripts/infer.py \
  --config configs/spot_centernet_resnet18.yaml \
  --checkpoint outputs/spot_centernet_resnet18/best.pt \
  --input photos \
  --output outputs/infer_resnet18

python3 scripts/infer.py \
  --config configs/spot_centernet_unet.yaml \
  --checkpoint outputs/spot_centernet_unet/best.pt \
  --input photos \
  --output outputs/infer_unet
```

#### 对单张图片推理

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet_dla34.yaml \
  --checkpoint outputs/spot_centernet_dla34/best.pt \
  --input photos/000001.jpg \
  --output outputs/infer_single
```

### 6.2 命令行参数

| 参数 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config` | 否 | `configs/spot_centernet.yaml` | 配置文件路径 |
| `--checkpoint` | **是** | — | 模型权重文件路径（`.pt`） |
| `--input` | **是** | — | 输入图像路径（单张文件或目录） |
| `--output` | 否 | `outputs/infer` | 推理结果输出目录 |
| `--score-threshold` | 否 | 配置文件中的值 | 检测置信度阈值 |
| `--topk` | 否 | 配置文件中的值 | 热力图 Top-K 候选数 |

### 6.3 推理输出

对于每张输入图像，会生成两个输出文件：

#### (1) 可视化图（`{name}_vis.jpg`）

三栏拼接图像：
- **左栏**：原始图像
- **中栏**：模型预测的热力图（JET 彩色映射）
- **右栏**：检测结果标注图（绿色十字标记 + 置信度分数）

#### (2) 检测结果 JSON（`{name}.json`）

```json
{
  "image": "photos/000001.jpg",
  "count": 36,
  "detections": [
    {
      "score": 0.9919,
      "x": 261.22,
      "y": 203.26,
      "class_id": 0
    },
    ...
  ]
}
```

每个检测点包含：
- `score`：置信度分数（0~1）
- `x`、`y`：光斑质心在原始图像坐标系中的坐标（像素级）
- `class_id`：类别 ID（固定为 0，即 `spot`）

---

## 7. 配置文件详解

项目提供三个预设配置文件，仅在骨干网络和输出目录上有区别，其余参数完全一致：

- `configs/spot_centernet_dla34.yaml`（DLA-34 骨干）
- `configs/spot_centernet_resnet18.yaml`（ResNet-18 骨干）
- `configs/spot_centernet_unet.yaml`（U-Net 骨干）

### 7.1 全局参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `experiment_name` | `spot_centernet_dla34` | 实验名称，用于区分不同实验 |
| `seed` | `42` | 全局随机种子，确保训练可复现 |

### 7.2 模型参数 `model`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `backbone` | `dla34` / `resnet18` / `unet` | 骨干网络类型 |
| `backbone_kwargs` | `{}` | 传给骨干网络构造函数的额外参数，例如 U-Net 的 `base_channels` |
| `neck_channels` | `128` | FPN 融合层通道数，越大表达能力越强但显存开销越大 |
| `head_channels` | `64` | 检测头内部卷积通道数 |

### 7.3 数据参数 `data`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `root` | `.` | 数据根目录 |
| `image_dir` | `photos` | 图像目录（相对 root） |
| `label_dir` | `labels_raw` | 标注目录（相对 root） |
| `split_dir` | `splits` | 划分文件目录（相对 root） |
| `train_split` | `train.txt` | 训练集 ID 列表文件名 |
| `val_split` | `val.txt` | 验证集 ID 列表文件名 |
| `val_ratio` | `0.25` | 自动划分时验证集比例 |
| `class_name` | `spot` | 目标类别名称（需与 Labelme 标注一致） |
| `spot_size_label` | `spot_size` | 每张图光斑尺寸标定线的标签名 |
| `spot_size_shape_type` | `line` | 光斑尺寸标定使用的 Labelme 形状类型 |
| `input_width` | `640` | 网络输入宽度（像素） |
| `input_height` | `384` | 网络输入高度（像素） |
| `down_ratio` | `4` | 输出特征图下采样比。输入 640×384 → 输出 160×96 |
| `max_objects` | `512` | 单张图最大编码目标数 |
| `point_box_width` | `36` | 未提供 `spot_size` 标定时的伪目标框宽度（回退用） |
| `point_box_height` | `36` | 未提供 `spot_size` 标定时的伪目标框高度（回退用） |
| `gaussian_min_overlap` | `0.7` | 高斯半径计算时的最小重叠要求 |
| `min_gaussian_radius` | `5` | 高斯半径下限（防止热力图过小） |
| `normalize_mean` | `[0.5, 0.5, 0.5]` | 图像归一化均值（RGB） |
| `normalize_std` | `[0.5, 0.5, 0.5]` | 图像归一化标准差（RGB） |

### 7.4 数据增强参数 `data.train_augment`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `brightness_gain` | `0.15` | 亮度扰动幅度，实际缩放范围 `[0.85, 1.15]` |
| `noise_std` | `0.01` | 高斯噪声标准差 |

### 7.5 训练参数 `train`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `batch_size` | `4` | 每批次样本数 |
| `num_workers` | `0` | DataLoader 并行加载进程数 |
| `epochs` | `120` | 总训练轮数 |
| `lr` | `0.001` | AdamW 优化器学习率 |
| `weight_decay` | `0.0001` | 权重衰减系数 |
| `log_interval` | `5` | 每隔多少 step 打印一次训练损失 |
| `save_dir` | `outputs/spot_centernet_dla34` | 模型输出保存目录 |
| `heatmap_loss_weight` | `1.0` | Heatmap Focal Loss 权重 |
| `offset_loss_weight` | `1.0` | Offset L1 Loss 权重 |

### 7.6 推理参数 `infer`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `score_threshold` | `0.3` | 检测置信度阈值，低于此值的检测点被过滤 |
| `topk` | `256` | 从热力图中提取的候选点上限 |
| `nms_kernel` | `9` | 局部极大值抑制的池化核大小 |

---

## 8. 完整使用流程

以下是从零开始的完整操作流程：

### 步骤 1：准备图像

将所有光斑图像放入 `photos/` 目录。

### 步骤 2：标注数据

使用 Labelme 对图像进行点标注：

```bash
labelme photos/ --output labels_raw/
```

在每个光斑中心标注一个 `spot` 类型的 point。

### 步骤 3：生成数据集划分

```bash
python3 scripts/make_splits.py --root .
```

### 步骤 4：训练模型

```bash
# 使用 DLA-34（推荐，精度更高）
python3 scripts/train.py --config configs/spot_centernet_dla34.yaml

# 或使用 ResNet-18（更轻量）
python3 scripts/train.py --config configs/spot_centernet_resnet18.yaml
```

### 步骤 5：推理检测

```bash
python3 scripts/infer.py \
  --config configs/spot_centernet_dla34.yaml \
  --checkpoint outputs/spot_centernet_dla34/best.pt \
  --input photos \
  --output outputs/infer_result
```

### 步骤 6：查看结果

- 可视化图：`outputs/infer_result/{name}_vis.jpg`
- 坐标数据：`outputs/infer_result/{name}.json`

---

## 9. 调参建议

### 漏检较多

1. **降低** `infer.score_threshold`（如 0.3 → 0.1），保留更多候选点
2. 优先检查 `spot_size` 标定是否偏小；标小会直接导致该图 GT 高斯核过窄
3. 若仍使用旧数据格式，再**增大** `data.point_box_width` 和 `data.point_box_height`，扩大高斯热力图覆盖范围
4. **增大** `data.min_gaussian_radius`，确保热力图不会过于稀疏

### 误检较多

1. **提高** `infer.score_threshold`（如 0.3 → 0.5），过滤低置信度检测
2. 优先检查 `spot_size` 标定是否偏大；标大会让该图 GT 高斯范围过宽
3. 若仍使用旧数据格式，再**减小** `data.point_box_width` 和 `data.point_box_height`，收窄高斯范围
4. **增加训练数据量**，标注更多样本

### 相邻光斑粘连

1. **减小** `infer.nms_kernel`（如 9 → 5 → 3），降低抑制强度
2. 优先检查 `spot_size` 标定线是否过长
3. 若仍使用旧数据格式，再**减小** `data.point_box_width` 和 `data.point_box_height`，让高斯核更紧凑

### 训练不稳定（loss 震荡）

1. **降低** `train.lr`（如 0.001 → 0.0005）
2. **增大** `train.batch_size`（需要足够显存）

### 目标数量多（>200）

1. **确认** `data.max_objects` ≥ 实际最大目标数 × 1.5
2. **确认** `infer.topk` ≥ 实际最大目标数 × 1.5

---

## 10. 模型架构详解

### 10.1 骨干网络（Backbone）

项目通过注册表机制支持多种骨干网络，当前已实现：

| 骨干网络 | 输出层数 | 各层通道数 | 各层步长 | 特点 |
| --- | --- | --- | --- | --- |
| **DLA-34** | 4 层 | [64, 128, 256, 512] | [4, 8, 16, 32] | 树状聚合结构，多尺度特征交互更充分 |
| **ResNet-18** | 4 层 | [64, 128, 256, 512] | [4, 8, 16, 32] | 经典残差网络，结构简单，推理速度快 |

### 10.2 颈部网络（Neck）—— FPN 融合

将 Backbone 输出的 4 层不同尺度特征通过 1×1 卷积统一到 `neck_channels`（默认 128）通道，然后上采样到最大分辨率后逐元素相加，最终通过 3×3 卷积输出融合特征。

### 10.3 检测头（Head）

两个并行的轻量检测头：

- **Heatmap Head**：输出 1 通道的热力图，表示每个位置存在光斑中心的概率
- **Offset Head**：输出 2 通道的偏移量 (Δx, Δy)，补偿下采样造成的量化误差

### 10.4 解码流程

1. 对 Heatmap 做 Sigmoid 激活
2. 使用 Max Pooling 进行局部极大值抑制（NMS），消除重复峰值
3. 取 Top-K 个得分最高的候选点
4. 加上对应位置的 Offset 偏移量
5. 通过逆仿射变换将特征图坐标映射回原始图像坐标
6. 按 `score_threshold` 过滤低置信度检测点

---

## 11. 核心源码说明

| 模块文件 | 功能说明 |
| --- | --- |
| `src/centernet_spot/config.py` | 加载 YAML 配置文件，返回字典 |
| `src/centernet_spot/data.py` | `SpotDataset` 数据集类：读取 Labelme 标注、仿射变换、高斯热力图编码、数据增强 |
| `src/centernet_spot/model.py` | 模型定义：`DLA34Backbone`、`ResNet18Backbone`、`FPNFusion`、`CenterNetHead`、`SpotCenterNet` |
| `src/centernet_spot/losses.py` | 损失函数：`focal_loss`（热力图分类）、`reg_l1_loss`（偏移量回归） |
| `src/centernet_spot/decode.py` | 推理解码：NMS、Top-K 选取、偏移量修正、坐标逆变换 |
| `src/centernet_spot/transforms.py` | 仿射变换工具：计算变换矩阵、坐标变换、逆变换 |
| `src/centernet_spot/split.py` | 数据划分：扫描标注文件、随机划分训练/验证集、读写划分文件 |
| `src/centernet_spot/utils.py` | 通用工具：随机种子设置、设备检测、目录创建、JSON 保存 |

---

## 12. 常见问题

### Q1：训练时提示 "No samples found for split=train"

**原因**：未生成数据集划分文件，且 `labels_raw/` 目录为空或没有标注文件。

**解决**：
1. 确认 `labels_raw/` 中有 `.json` 标注文件
2. 运行 `python3 scripts/make_splits.py --root .`

### Q2：推理时提示 "No images found"

**原因**：`--input` 参数指定的路径不存在，或者目录中没有支持的图片格式。

**解决**：确认路径正确，图片格式为 `.jpg`、`.jpeg`、`.png` 或 `.bmp`。

### Q3：训练 loss 不下降

**可能原因**：
- 标注数据太少（建议至少 20+ 张）
- 学习率过大
- 标注质量有误（label 名不是 `spot`，或 shape_type 不是 `point`）

### Q4：如何添加新的骨干网络？

在 `src/centernet_spot/model.py` 中使用 `@register_backbone("name")` 装饰器注册新的 Backbone 类，该类需要：
1. 继承 `nn.Module`
2. 提供 `out_channels: List[int]` 类属性（各层输出通道数）
3. `forward()` 返回多尺度特征列表 `List[torch.Tensor]`

然后在配置文件中设置 `model.backbone: name` 即可使用。

### Q5：GPU 显存不足怎么办？

1. 减小 `train.batch_size`（如 4 → 2 → 1）
2. 减小 `data.input_width` 和 `data.input_height`
3. 使用更轻量的骨干网络（`resnet18`）
4. 减小 `model.neck_channels` 和 `model.head_channels`
