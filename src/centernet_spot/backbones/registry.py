from __future__ import annotations

from torch import nn


_BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {}


def register_backbone(name: str):
    """装饰器：将 backbone 类注册到全局表。"""
    def wrapper(cls):
        _BACKBONE_REGISTRY[name] = cls
        return cls
    return wrapper


def build_backbone(name: str, **kwargs) -> nn.Module:
    """根据名字构建 backbone，返回的模块需提供 out_channels 属性。"""
    if name.startswith("timm:"):
        from .timm_backbone import TimmBackbone

        model_name = name.split(":", 1)[1].strip()
        if not model_name:
            raise ValueError("Timm backbone 必须开头是 'timm:'.")
        return TimmBackbone(model_name=model_name, **kwargs)

    if name not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        extra = " and any timm model via 'timm:<model_name>'"
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}{extra}")
    return _BACKBONE_REGISTRY[name](**kwargs)
