from __future__ import annotations

from typing import List

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .registry import register_backbone
from .utils import resolve_torchvision_weights


@register_backbone("mobilenetv3_large")
class MobileNetV3LargeBackbone(nn.Module):
    """MobileNetV3-Large backbone，输出 4 层特征 (stride 4/8/16/32)。"""

    out_channels: List[int] = [24, 40, 112, 960]
    _feature_indices = (3, 6, 12, 16)

    def __init__(self, pretrained: bool = True, weights: str | None = None, **_kw) -> None:
        super().__init__()
        resolved_weights = resolve_torchvision_weights(
            pretrained=pretrained,
            weights=weights,
            enum_cls=MobileNet_V3_Large_Weights,
            default_weight=MobileNet_V3_Large_Weights.DEFAULT,
        )
        network = mobilenet_v3_large(weights=resolved_weights)
        self.features = network.features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self._feature_indices:
                feats.append(x)
        if len(feats) != len(self.out_channels):
            raise RuntimeError(
                f"Expected {len(self.out_channels)} MobileNetV3 feature maps, got {len(feats)}."
            )
        return feats
