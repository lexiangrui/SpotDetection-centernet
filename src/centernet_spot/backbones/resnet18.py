from __future__ import annotations

from typing import List

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .registry import register_backbone
from .utils import resolve_torchvision_weights


@register_backbone("resnet18")
class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone，输出 4 层特征 (stride 4/8/16/32)，channels [64,128,256,512]。"""

    out_channels: List[int] = [64, 128, 256, 512]

    def __init__(self, pretrained: bool = False, weights: str | None = None, **_kw) -> None:
        super().__init__()
        resolved_weights = resolve_torchvision_weights(
            pretrained=pretrained,
            weights=weights,
            enum_cls=ResNet18_Weights,
            default_weight=ResNet18_Weights.DEFAULT,
        )
        network = resnet18(weights=resolved_weights)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
