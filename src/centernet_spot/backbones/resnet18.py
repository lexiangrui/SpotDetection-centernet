from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone，输出 4 层特征 (stride 4/8/16/32)。"""

    out_channels: list[int] = [64, 128, 256, 512]

    def __init__(self, load_pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if load_pretrained else None
        network = resnet18(weights=weights)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
