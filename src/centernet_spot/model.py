from __future__ import annotations

from typing import List

import torch
from torch.hub import load_state_dict_from_url
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    resnet18,
)

from .utils import sigmoid_heatmap


# ============================================================
#  Backbone 注册表
# ============================================================
_BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {}
_DLA34_IMAGENET_URL = "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"


def _resolve_torchvision_weights(pretrained: bool, weights: str | None, enum_cls, default_weight):
    if not pretrained:
        return None
    if weights is None or str(weights).lower() == "default":
        return default_weight
    try:
        return enum_cls[str(weights)]
    except KeyError as exc:
        available = ", ".join(weight.name for weight in enum_cls)
        raise ValueError(f"Unknown weights '{weights}'. Available: {available}") from exc


def register_backbone(name: str):
    """装饰器：将 backbone 类注册到全局表。"""
    def wrapper(cls):
        _BACKBONE_REGISTRY[name] = cls
        return cls
    return wrapper


def build_backbone(name: str, **kwargs) -> nn.Module:
    """根据名字构建 backbone，返回的模块需提供 out_channels 属性。"""
    if name not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}")
    return _BACKBONE_REGISTRY[name](**kwargs)


# ============================================================
#  DLA-34 Backbone
# ============================================================

class _DLABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is None:
            residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class _Root(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, residual: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(torch.cat(inputs, dim=1)))
        if self.residual:
            out += inputs[0]
        return self.relu(out)


class _Tree(nn.Module):
    def __init__(self, levels, block, in_ch, out_ch, stride=1, level_root=False,
                 root_dim=0, root_ks=1, dilation=1, root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_ch
        if level_root:
            root_dim += in_ch
        if levels == 1:
            self.tree1 = block(in_ch, out_ch, stride=stride, dilation=dilation)
            self.tree2 = block(out_ch, out_ch, stride=1, dilation=dilation)
        else:
            self.tree1 = _Tree(levels - 1, block, in_ch, out_ch, stride=stride,
                               root_dim=0, root_ks=root_ks, dilation=dilation, root_residual=root_residual)
            self.tree2 = _Tree(levels - 1, block, out_ch, out_ch,
                               root_dim=root_dim + out_ch, root_ks=root_ks,
                               dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = _Root(root_dim, out_ch, root_ks, root_residual)
        self.level_root = level_root
        self.levels = levels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
                        if in_ch != out_ch else None)

    def forward(self, x, residual=None, children=None):
        if children is None:
            children = []
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        children.append(x1)
        return self.tree2(x1, children=children)


@register_backbone("dla34")
class DLA34Backbone(nn.Module):
    """DLA-34 backbone，输出 4 层特征 (stride 4/8/16/32)，channels [64,128,256,512]。"""

    out_channels: List[int] = [64, 128, 256, 512]

    def __init__(self, pretrained: bool = False, weights: str | None = None, progress: bool = True, **_kw) -> None:
        super().__init__()
        channels = [16, 32, 64, 128, 256, 512]
        levels = [1, 1, 1, 2, 2, 1]
        block = _DLABasicBlock
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv(channels[0], channels[1], levels[1], stride=2)
        self.level2 = _Tree(levels[2], block, channels[1], channels[2], stride=2)
        self.level3 = _Tree(levels[3], block, channels[2], channels[3], stride=2, level_root=True)
        self.level4 = _Tree(levels[4], block, channels[3], channels[4], stride=2, level_root=True)
        self.level5 = _Tree(levels[5], block, channels[4], channels[5], stride=2, level_root=True)

        if pretrained:
            weight_name = "imagenet" if weights is None or str(weights).lower() == "default" else str(weights).lower()
            if weight_name != "imagenet":
                raise ValueError("DLA34 official pretrained weights only provide 'imagenet'.")

            state_dict = load_state_dict_from_url(_DLA34_IMAGENET_URL, progress=progress, check_hash=True)
            incompatible = self.load_state_dict(state_dict, strict=False)
            unexpected = set(incompatible.unexpected_keys) - {"fc.weight", "fc.bias"}
            if incompatible.missing_keys or unexpected:
                raise RuntimeError(
                    "Failed to load official DLA34 pretrained weights cleanly. "
                    f"missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}"
                )

    @staticmethod
    def _make_conv(in_c, out_c, n, stride=1):
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, 3,
                                 stride=stride if i == 0 else 1, padding=1, bias=False),
                       nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y: List[torch.Tensor] = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f"level{i}")(x)
            y.append(x)
        # 返回 stride 4/8/16/32 的特征
        return y[2:]


# ============================================================
@register_backbone("resnet18")
class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone，输出 4 层特征 (stride 4/8/16/32)，channels [64,128,256,512]。"""

    out_channels: List[int] = [64, 128, 256, 512]

    def __init__(self, pretrained: bool = False, weights: str | None = None, **_kw) -> None:
        super().__init__()
        resolved_weights = _resolve_torchvision_weights(
            pretrained=pretrained,
            weights=weights,
            enum_cls=ResNet18_Weights,
            default_weight=ResNet18_Weights.DEFAULT,
        )
        network = resnet18(weights=resolved_weights)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1  # stride 4
        self.layer2 = network.layer2  # stride 8
        self.layer3 = network.layer3  # stride 16
        self.layer4 = network.layer4  # stride 32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        return [c2, c3, c4, c5]


# ============================================================
#  MobileNetV3 Backbone
# ============================================================

@register_backbone("mobilenetv3_large")
class MobileNetV3LargeBackbone(nn.Module):
    """MobileNetV3-Large backbone，输出 4 层特征 (stride 4/8/16/32)。"""

    out_channels: List[int] = [24, 40, 112, 960]
    _feature_indices = (3, 6, 12, 16)

    def __init__(self, pretrained: bool = True, weights: str | None = None, **_kw) -> None:
        super().__init__()
        resolved_weights = _resolve_torchvision_weights(
            pretrained=pretrained,
            weights=weights,
            enum_cls=MobileNet_V3_Large_Weights,
            default_weight=MobileNet_V3_Large_Weights.DEFAULT,
        )
        network = mobilenet_v3_large(weights=resolved_weights)
        self.features = network.features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self._feature_indices:
                feats.append(x)
        if len(feats) != len(self.out_channels):
            raise RuntimeError(
                f"Expected {len(self.out_channels)} MobileNetV3 feature maps, got {len(feats)}."
            )
        return feats


# ============================================================
#  U-Net Backbone
# ============================================================

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


@register_backbone("unet")
class UNetBackbone(nn.Module):
    """U-Net backbone，输出 4 层特征 (stride 4/8/16/32)。"""

    def __init__(self, base_channels: int = 32, pretrained: bool = False, **_kw) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("UNet has no official pretrained weights in this project. Set pretrained=false.")
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.out_channels: List[int] = [c3, c4, c5, c5]

        self.stem = _DoubleConv(3, c1)       # stride 1
        self.down1 = _DownBlock(c1, c2)      # stride 2
        self.down2 = _DownBlock(c2, c3)      # stride 4
        self.down3 = _DownBlock(c3, c4)      # stride 8
        self.down4 = _DownBlock(c4, c5)      # stride 16
        self.bottleneck = _DownBlock(c5, c5) # stride 32

        self.up3 = _UpBlock(c5, c5, c5)      # stride 16
        self.up2 = _UpBlock(c5, c4, c4)      # stride 8
        self.up1 = _UpBlock(c4, c3, c3)      # stride 4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.bottleneck(x4)

        d3 = self.up3(x5, x4)
        d2 = self.up2(d3, x3)
        d1 = self.up1(d2, x2)
        return [d1, d2, d3, x5]


# ============================================================
#  Neck & Head
# ============================================================

class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPNFusion(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for in_c in in_channels_list
        ])
        self.deep_context = _DepthwiseSeparableBlock(out_channels, out_channels)
        self.refine = nn.ModuleList([
            _DepthwiseSeparableBlock(out_channels * 2, out_channels)
            for _ in range(len(in_channels_list) - 1)
        ])
        self.output = _DepthwiseSeparableBlock(out_channels, out_channels)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pyramid = [lat(feat) for feat, lat in zip(features, self.lateral)]
        x = self.deep_context(pyramid[-1])

        for level in range(len(pyramid) - 2, -1, -1):
            upsampled = F.interpolate(x, size=pyramid[level].shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine[level](torch.cat([pyramid[level], upsampled], dim=1))

        return self.output(x)


class CenterNetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, head_channels: int,
                 final_bias: float | None = None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, out_channels, 1, bias=True),
        )
        if final_bias is not None:
            nn.init.constant_(self.layers[-1].bias, final_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ============================================================
#  SpotCenterNet: 通过 config 选择 backbone
# ============================================================

class SpotCenterNet(nn.Module):
    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {}) if cfg else {}
        backbone_name = str(model_cfg.get("backbone", "dla34"))
        neck_channels = int(model_cfg.get("neck_channels", 128))
        head_channels = int(model_cfg.get("head_channels", 64))
        backbone_kwargs = dict(model_cfg.get("backbone_kwargs", {}))

        self.backbone = build_backbone(backbone_name, **backbone_kwargs)
        fpn_in_channels: List[int] = self.backbone.out_channels  # type: ignore[attr-defined]

        self.neck = FPNFusion(fpn_in_channels, neck_channels)
        self.hm_head = CenterNetHead(neck_channels, 1, head_channels, final_bias=-2.19)
        self.reg_head = CenterNetHead(neck_channels, 2, head_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        feat = self.neck(features)
        heatmap = sigmoid_heatmap(self.hm_head(feat))
        return {
            "heatmap": heatmap,
            "reg": self.reg_head(feat),
        }
