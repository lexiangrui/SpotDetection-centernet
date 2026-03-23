"""Backbone 注册表与实现。

导入此包时自动注册所有 backbone。
"""
from .registry import build_backbone, register_backbone

from . import dla34 as _dla34      # noqa: F401
from . import mobilenetv3 as _mv3  # noqa: F401
from . import resnet18 as _r18     # noqa: F401
from . import timm_backbone as _timm  # noqa: F401

__all__ = ["build_backbone", "register_backbone"]
