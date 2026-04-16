"""CenterNet baseline for spot detection."""

from .model import SpotCenterNet, build_model_from_config

__all__ = ["SpotCenterNet", "build_model_from_config"]
