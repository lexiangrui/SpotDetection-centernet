from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


def _fill_up_weights(up: nn.ConvTranspose2d) -> None:
    weight = up.weight.data
    factor = math.ceil(weight.size(2) / 2)
    center = (2 * factor - 1 - factor % 2) / (2.0 * factor)
    for i in range(weight.size(2)):
        for j in range(weight.size(3)):
            weight[0, 0, i, j] = (1 - abs(i / factor - center)) * (1 - abs(j / factor - center))
    for channel in range(1, weight.size(0)):
        weight[channel, 0, :, :] = weight[0, 0, :, :]


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpsampleTranspose(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False)
        _fill_up_weights(up)
        self.block = nn.Sequential(
            up,
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetCenterNetDecoder(nn.Module):
    """ResNet-18 decoder using 3-stage conv + deconv upsampling."""

    out_channels: int = 64

    def __init__(self, in_channels_list: Sequence[int]) -> None:
        super().__init__()
        deconv_channels = [256, 128, 64]
        layers: list[nn.Module] = []
        prev_channels = int(in_channels_list[-1])
        for out_channels in deconv_channels:
            layers.append(
                nn.Sequential(
                    _ConvBNReLU(prev_channels, out_channels),
                    _UpsampleTranspose(out_channels),
                )
            )
            prev_channels = out_channels
        self.decoder = nn.Sequential(*layers)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decoder(features[-1])
