from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from .registry import register_backbone

_DLA34_IMAGENET_URL = "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"


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
        return y[2:]
