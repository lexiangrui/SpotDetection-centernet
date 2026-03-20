from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def build_resize_pad_transform(orig_w: int, orig_h: int, dst_w: int, dst_h: int) -> dict[str, float | int]:
    scale = min(dst_w / float(orig_w), dst_h / float(orig_h))
    resized_w = max(int(round(orig_w * scale)), 1)
    resized_h = max(int(round(orig_h * scale)), 1)
    pad_left = max((dst_w - resized_w) // 2, 0)
    pad_top = max((dst_h - resized_h) // 2, 0)
    pad_right = max(dst_w - resized_w - pad_left, 0)
    pad_bottom = max(dst_h - resized_h - pad_top, 0)
    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "dst_w": dst_w,
        "dst_h": dst_h,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "scale_x": resized_w / float(orig_w),
        "scale_y": resized_h / float(orig_h),
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def resize_and_pad_image(
    image: np.ndarray,
    output_size: Tuple[int, int],
    pad_value: int | tuple[int, int, int] = 0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    dst_w, dst_h = output_size
    transform = build_resize_pad_transform(image.shape[1], image.shape[0], dst_w, dst_h)
    resized = cv2.resize(
        image,
        (int(transform["resized_w"]), int(transform["resized_h"])),
        interpolation=cv2.INTER_LINEAR,
    )

    if image.ndim == 2:
        canvas = np.full((dst_h, dst_w), pad_value, dtype=image.dtype)
    else:
        channels = image.shape[2]
        canvas = np.full((dst_h, dst_w, channels), pad_value, dtype=image.dtype)

    left = int(transform["pad_left"])
    top = int(transform["pad_top"])
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas, transform


def transform_point(pt: Iterable[float], transform: dict[str, float | int]) -> np.ndarray:
    x, y = float(pt[0]), float(pt[1])
    return np.array(
        [
            x * float(transform["scale_x"]) + float(transform["pad_left"]),
            y * float(transform["scale_y"]) + float(transform["pad_top"]),
        ],
        dtype=np.float32,
    )


def inverse_transform_points(coords: np.ndarray, transform: dict[str, float | int]) -> np.ndarray:
    target_coords = np.zeros(coords.shape, dtype=np.float32)
    target_coords[:, 0] = (coords[:, 0] - float(transform["pad_left"])) / max(float(transform["scale_x"]), 1e-8)
    target_coords[:, 1] = (coords[:, 1] - float(transform["pad_top"])) / max(float(transform["scale_y"]), 1e-8)
    return target_coords
