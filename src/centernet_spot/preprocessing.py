"""统一的图像预处理函数，供推理和 RKNN 数据集制备共用。"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .transforms import resize_and_pad_image
from .utils import normalize_rgb_image


def preprocess_image(image_bgr: np.ndarray, cfg: dict) -> torch.Tensor:
    """BGR 图像 -> 归一化后的 [1, 3, H, W] float32 tensor。"""
    input_w = int(cfg["data"]["input_width"])
    input_h = int(cfg["data"]["input_height"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized, _ = resize_and_pad_image(image_rgb, (input_w, input_h))
    image_norm = normalize_rgb_image(image_resized, cfg)
    return torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()


def preprocess_image_numpy(image_bgr: np.ndarray, cfg: dict,
                           input_w: int | None = None,
                           input_h: int | None = None) -> np.ndarray:
    """BGR 图像 -> 归一化后的 [1, 3, H, W] float32 ndarray（用于 RKNN 等非 PyTorch 场景）。"""
    if input_w is None:
        input_w = int(cfg["data"]["input_width"])
    if input_h is None:
        input_h = int(cfg["data"]["input_height"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized, _ = resize_and_pad_image(image_rgb, (input_w, input_h))
    image_norm = normalize_rgb_image(image_resized, cfg)
    return np.expand_dims(image_norm.transpose(2, 0, 1).astype(np.float32), axis=0)
