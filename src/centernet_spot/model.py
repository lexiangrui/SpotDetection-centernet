from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .backbones import ResNet18Backbone
from .head import CenterNetHead
from .neck import ResNetCenterNetDecoder

HEAD_CHANNELS = 48


def heatmap_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.sigmoid()


class SpotCenterNet(nn.Module):
    def __init__(self, *, load_pretrained_backbone: bool = True, dropout_p: float = 0.0) -> None:
        super().__init__()
        dropout_p = float(dropout_p)
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}.")

        self.backbone = ResNet18Backbone(load_pretrained=load_pretrained_backbone)
        self.decoder = ResNetCenterNetDecoder(self.backbone.out_channels)
        feat_channels = int(self.decoder.out_channels)
        self.dropout_p = dropout_p
        self.feature_dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        self.hm_head = CenterNetHead(feat_channels, 1, HEAD_CHANNELS, final_bias=-2.19)
        self.reg_head = CenterNetHead(feat_channels, 2, HEAD_CHANNELS)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        feat = self.feature_dropout(self.decoder(features))
        heatmap_logits = self.hm_head(feat)
        return {
            "heatmap_logits": heatmap_logits,
            "reg": self.reg_head(feat),
        }


def build_model_from_config(
    cfg: Mapping[str, Any],
    *,
    load_pretrained_backbone: bool = True,
) -> SpotCenterNet:
    model_cfg = cfg.get("model", {})
    dropout_p = float(model_cfg.get("dropout_p", 0.0))
    return SpotCenterNet(
        load_pretrained_backbone=load_pretrained_backbone,
        dropout_p=dropout_p,
    )
