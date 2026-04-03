from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def get_input_normalization(_cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    return IMAGENET_MEAN.copy(), IMAGENET_STD.copy()


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
