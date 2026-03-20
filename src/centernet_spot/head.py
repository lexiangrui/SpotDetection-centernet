from __future__ import annotations

import torch
from torch import nn


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
