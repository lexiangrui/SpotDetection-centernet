from __future__ import annotations

from typing import List

import torch
from torch import nn

from .backbones import build_backbone
from .head import CenterNetHead
from .neck import FPNFusion


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
        heatmap = self.hm_head(feat).sigmoid()
        return {
            "heatmap": heatmap,
            "reg": self.reg_head(feat),
        }
