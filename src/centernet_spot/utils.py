from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .backbones.timm_backbone import get_timm_model_data_config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


@lru_cache(maxsize=64)
def _get_timm_normalization(model_name: str, in_chans: int = 3) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = get_timm_model_data_config(model_name, in_chans=in_chans)
    mean = np.asarray(data_cfg.get("mean", IMAGENET_MEAN.tolist()), dtype=np.float32)
    std = np.asarray(data_cfg.get("std", IMAGENET_STD.tolist()), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError(
            f"timm backbone '{model_name}' 返回的 mean/std 不是 3 通道配置: mean={mean}, std={std}"
        )
    if np.any(std <= 0):
        raise ValueError(f"timm backbone '{model_name}' 返回了非正 std: {std}")
    return mean, std


@lru_cache(maxsize=64)
def _get_timm_normalization_with_cfg_cache(
    model_name: str,
    in_chans: int = 3,
    pretrained_cfg: str | None = None,
    pretrained_cfg_overlay_json: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pretrained_cfg_overlay = (
        json.loads(pretrained_cfg_overlay_json) if pretrained_cfg_overlay_json is not None else None
    )
    build_kwargs: dict[str, Any] = {"in_chans": in_chans}
    if pretrained_cfg is not None:
        build_kwargs["pretrained_cfg"] = pretrained_cfg
    if pretrained_cfg_overlay is not None:
        build_kwargs["pretrained_cfg_overlay"] = pretrained_cfg_overlay

    data_cfg = get_timm_model_data_config(model_name, **build_kwargs)
    mean = np.asarray(data_cfg.get("mean", IMAGENET_MEAN.tolist()), dtype=np.float32)
    std = np.asarray(data_cfg.get("std", IMAGENET_STD.tolist()), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError(
            f"timm backbone '{model_name}' returned invalid 3-channel mean/std: mean={mean}, std={std}"
        )
    if np.any(std <= 0):
        raise ValueError(f"timm backbone '{model_name}' returned non-positive std: {std}")
    return mean, std, data_cfg


def _resolve_timm_normalization(
    model_name: str,
    in_chans: int = 3,
    pretrained_cfg: str | None = None,
    pretrained_cfg_overlay: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    overlay_json = None
    if pretrained_cfg_overlay is not None:
        overlay_json = json.dumps(pretrained_cfg_overlay, sort_keys=True)
    mean, std, data_cfg = _get_timm_normalization_with_cfg_cache(
        model_name=model_name,
        in_chans=in_chans,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay_json=overlay_json,
    )
    return mean.copy(), std.copy(), dict(data_cfg)


def resolve_input_normalization(cfg: dict | None = None) -> dict[str, Any]:
    model_cfg = cfg.get("model", {}) if cfg else {}
    norm_cfg = model_cfg.get("input_normalization", {})
    mean = np.asarray(norm_cfg.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.asarray(norm_cfg.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)

    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("input_normalization.mean/std must each contain 3 values.")
    if np.any(std <= 0):
        raise ValueError("input_normalization.std must be positive.")

    if norm_cfg:
        return {
            "mean": mean,
            "std": std,
            "source": "config",
            "source_detail": "model.input_normalization",
            "timm_data_config": None,
        }

    backbone_name = str(model_cfg.get("backbone", ""))
    backbone_kwargs = dict(model_cfg.get("backbone_kwargs", {}))
    pretrained = bool(backbone_kwargs.get("pretrained", False))

    if backbone_name.startswith("timm:") and pretrained:
        model_name = backbone_name.split(":", 1)[1].strip()
        in_chans = int(backbone_kwargs.get("in_chans", 3))
        pretrained_cfg = backbone_kwargs.get("pretrained_cfg")
        pretrained_cfg_overlay = backbone_kwargs.get("pretrained_cfg_overlay")
        mean, std, data_cfg = _resolve_timm_normalization(
            model_name,
            in_chans=in_chans,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
        )
        return {
            "mean": mean,
            "std": std,
            "source": "timm_default",
            "source_detail": f"timm pretrained_cfg for '{model_name}'",
            "timm_data_config": data_cfg,
        }

    if backbone_name in {"mobilenetv3_large", "resnet18", "dla34"} and pretrained:
        return {
            "mean": IMAGENET_MEAN.copy(),
            "std": IMAGENET_STD.copy(),
            "source": "imagenet_default",
            "source_detail": "built-in pretrained backbone fallback",
            "timm_data_config": None,
        }

    return {
        "mean": mean,
        "std": std,
        "source": "identity_default",
        "source_detail": "no explicit normalization and no pretrained default",
        "timm_data_config": None,
    }


def get_input_normalization(cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    resolved = resolve_input_normalization(cfg)
    return resolved["mean"].copy(), resolved["std"].copy()


def describe_preprocessing(
    cfg: dict | None = None,
    *,
    input_w: int | None = None,
    input_h: int | None = None,
) -> dict[str, Any]:
    data_cfg = cfg.get("data", {}) if cfg else {}
    target_w = int(input_w if input_w is not None else data_cfg.get("input_width", 0))
    target_h = int(input_h if input_h is not None else data_cfg.get("input_height", 0))
    norm = resolve_input_normalization(cfg)
    mean = norm["mean"]
    std = norm["std"]

    description: dict[str, Any] = {
        "color_space": "BGR->RGB",
        "resize": {
            "type": "aspect_ratio_preserving_resize_and_pad",
            "target_width": target_w,
            "target_height": target_h,
            "interpolation": "cv2.INTER_LINEAR",
            "pad_value": 0,
        },
        "normalize": {
            "type": "float32_div_255_then_channelwise_normalize",
            "formula": "x = (x / 255.0 - mean) / std",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "source": norm["source"],
            "source_detail": norm["source_detail"],
        },
    }

    timm_data_config = norm.get("timm_data_config")
    if timm_data_config:
        description["normalize"]["timm_data_config"] = {
            key: _to_jsonable(timm_data_config.get(key))
            for key in ("input_size", "test_input_size", "mean", "std", "interpolation", "crop_pct")
            if key in timm_data_config
        }

    return description


def print_preprocessing_summary(
    cfg: dict | None = None,
    *,
    input_w: int | None = None,
    input_h: int | None = None,
    prefix: str = "preprocessing",
) -> None:
    payload = describe_preprocessing(cfg, input_w=input_w, input_h=input_h)
    print(json.dumps({prefix: payload}, ensure_ascii=False))


def normalize_rgb_image(image: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    mean, std = get_input_normalization(cfg)
    image_f = image.astype(np.float32) / 255.0
    return (image_f - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)


def denormalize_image(image: np.ndarray, cfg: dict | None = None, channel_first: bool = False) -> np.ndarray:
    mean, std = get_input_normalization(cfg)
    out = image.astype(np.float32).copy()
    if channel_first:
        out = out * std[:, None, None] + mean[:, None, None]
    else:
        out = out * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)
