from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


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
