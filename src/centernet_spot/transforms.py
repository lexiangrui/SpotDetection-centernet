from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def get_dir(src_point: Iterable[float], rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0.0, 0.0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return np.array(src_result, dtype=np.float32)


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: np.ndarray | None = None,
    inv: bool = False,
) -> np.ndarray:
    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)

    if not isinstance(scale, np.ndarray):
        scale = np.array(scale, dtype=np.float32)

    src_w, src_h = scale[0], scale[1]
    dst_w, dst_h = output_size
    rot_rad = np.pi * rot / 180.0
    src_dir = get_dir([0, -0.5 * src_h], rot_rad)
    dst_dir = np.array([0, -0.5 * dst_h], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_transform(pt: Iterable[float], transform: np.ndarray) -> np.ndarray:
    pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    new_pt = np.dot(transform, pt)
    return new_pt[:2]


def transform_preds(coords: np.ndarray, center: np.ndarray, scale: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    target_coords = np.zeros(coords.shape, dtype=np.float32)
    trans = get_affine_transform(center, scale, 0, output_size, inv=True)
    for idx, coord in enumerate(coords):
        target_coords[idx, 0:2] = affine_transform(coord[0:2], trans)
    return target_coords
