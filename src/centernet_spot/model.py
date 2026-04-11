from __future__ import annotations

import torch
from torch import nn

from .backbones import ResNet18Backbone
from .head import CenterNetHead
from .neck import ResNetCenterNetDecoder

HEAD_CHANNELS = 48


def heatmap_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.sigmoid()


class SpotCenterNet(nn.Module):
    def __init__(self, *, load_pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(load_pretrained=load_pretrained_backbone)
        self.decoder = ResNetCenterNetDecoder(self.backbone.out_channels)
        feat_channels = int(self.decoder.out_channels)

        self.hm_head = CenterNetHead(feat_channels, 1, HEAD_CHANNELS, final_bias=-2.19)
        self.reg_head = CenterNetHead(feat_channels, 2, HEAD_CHANNELS)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        feat = self.decoder(features)
        heatmap_logits = self.hm_head(feat)
        return {
            "heatmap_logits": heatmap_logits,
            "reg": self.reg_head(feat),
        }
