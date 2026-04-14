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


class _PointwiseBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
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


class _DecoderStage(nn.Module):
    def __init__(
        self,
        top_in_channels: int,
        skip_in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.topdown = _ConvBNReLU(top_in_channels, out_channels)
        self.upsample = _UpsampleTranspose(out_channels)
        self.lateral = _PointwiseBN(skip_in_channels, out_channels)
        self.refine = _ConvBNReLU(out_channels, out_channels)

    def forward(self, top: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.topdown(top)
        x = self.upsample(x)
        x = x + self.lateral(skip)
        return self.refine(x)


class ResNetCenterNetDecoder(nn.Module):
    """ResNet-18 decoder with full top-down skip fusion."""

    out_channels: int = 64

    def __init__(self, in_channels_list: Sequence[int]) -> None:
        super().__init__()
        if len(in_channels_list) < 4:
            raise ValueError("ResNetCenterNetDecoder expects 4 backbone feature levels (C2-C5).")

        c2_channels, c3_channels, c4_channels, c5_channels = [int(ch) for ch in in_channels_list[-4:]]
        self.stages = nn.ModuleList([
            _DecoderStage(c5_channels, c4_channels, 256),
            _DecoderStage(256, c3_channels, 128),
            _DecoderStage(128, c2_channels, 64),
        ])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = features[-4:]
        x = c5
        for stage, skip in zip(self.stages, (c4, c3, c2)):
            x = stage(x, skip)
        return x
