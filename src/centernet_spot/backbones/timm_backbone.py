from __future__ import annotations

from typing import Any, List, Sequence

import torch
from torch import nn

try:
    import timm
    from timm.data import resolve_model_data_config
except ImportError as exc:  # pragma: no cover - import guard
    timm = None
    resolve_model_data_config = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


def _ensure_timm_available() -> None:
    if timm is None:
        raise ImportError("先运行 pip install timm") from _TIMM_IMPORT_ERROR


def get_timm_model_data_config(model_name: str, **kwargs) -> dict[str, Any]:
    """读取 timm 模型默认的数据配置（mean/std/input_size 等）。"""
    _ensure_timm_available()
    build_kwargs = dict(kwargs)
    build_kwargs.setdefault("pretrained", False)
    build_kwargs.pop("features_only", None)
    build_kwargs.pop("out_indices", None)
    model = timm.create_model(model_name, **build_kwargs)
    if resolve_model_data_config is not None:
        return dict(resolve_model_data_config(model))

    default_cfg = getattr(model, "default_cfg", None) or getattr(model, "pretrained_cfg", None)
    if default_cfg is None:
        raise ValueError(f"timm backbone '{model_name}' 没有可用的默认数据配置")
    return dict(default_cfg)


class TimmBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_indices: Sequence[int] | None = None,
        in_chans: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        _ensure_timm_available()

        timm_kwargs = dict(kwargs)
        timm_kwargs.setdefault("features_only", True)
        timm_kwargs.setdefault("pretrained", pretrained)
        timm_kwargs.setdefault("in_chans", in_chans)
        if out_indices is not None:
            timm_kwargs["out_indices"] = tuple(int(idx) for idx in out_indices)

        try:
            self.model = timm.create_model(model_name, **timm_kwargs)
        except TypeError as exc:
            raise ValueError(
                f"timm库backbone加载失败：'{model_name}'。"
                "请确保该模型支持 features_only=True 且提供的 kwargs 是有效的"
            ) from exc

        feature_info = getattr(self.model, "feature_info", None)
        if feature_info is None:
            raise ValueError(f"timm backbone '{model_name}' 没有 feature_info 属性，无法获取输出通道数")

        channels = feature_info.channels()
        if not channels:
            raise ValueError(f"timm backbone '{model_name}' 没有返回特征通道数")

        self.model_name = model_name
        self.out_channels: List[int] = [int(ch) for ch in channels]
        self.reductions: List[int] = [int(v) for v in feature_info.reduction()]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.model(x)
        if isinstance(features, tuple):
            features = list(features)
        if not isinstance(features, list) or not features:
            raise RuntimeError(
                f"timm backbone '{self.model_name}' 返回了无效的特征: {type(features)!r}"
            )
        return features
