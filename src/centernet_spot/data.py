from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_resize_pad_transform, resize_and_pad_image, transform_point
from .utils import normalize_rgb_image

POINT_LABEL = "spot"


def gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], diameter: float) -> None:
    radius = max(float(diameter) / 2.0, 1e-6)
    radius_int = int(np.ceil(radius))
    size = 2 * radius_int + 1
    gaussian = gaussian2d((size, size), sigma=max(float(diameter) / 6.0, 1e-3))
    x, y = center
    height, width = heatmap.shape

    left = min(x, radius_int)
    right = min(width - x, radius_int + 1)
    top = min(y, radius_int)
    bottom = min(height - y, radius_int + 1)

    if min(left, right, top, bottom) <= 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius_int - top : radius_int + bottom,
        radius_int - left : radius_int + right,
    ]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


class SpotDataset(Dataset):
    def __init__(self, cfg: Dict, split_name: str, training: bool) -> None:
        self.cfg = cfg
        self.training = training

        data_cfg = cfg["data"]
        self.root = Path(data_cfg["root"])
        self.input_w = int(data_cfg["input_width"])
        self.input_h = int(data_cfg["input_height"])
        self.down_ratio = int(data_cfg["down_ratio"])
        self.out_w = self.input_w // self.down_ratio
        self.out_h = self.input_h // self.down_ratio
        self.max_objects = int(data_cfg["max_objects"])
        self.heatmap_diameter_out = float(data_cfg.get("heatmap_diameter_out", 6.0))
        self.augment_cfg = data_cfg.get("train_augment", {})

        image_dir_key = f"{split_name}_image_dir"
        if image_dir_key not in data_cfg:
            raise KeyError(f"Missing data.{image_dir_key}")
        self.image_dir = self.root / data_cfg[image_dir_key]
        self.sample_ids = sorted(p.stem for p in self.image_dir.glob("*.json"))

        if not self.sample_ids:
            raise RuntimeError(f"No samples found for split={split_name}.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_labelme(
        self, sample_id: str
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], Dict]:
        label_path = self.image_dir / f"{sample_id}.json"
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        raw_image_path = str(ann.get("imagePath", "")).strip()
        image_path = (label_path.parent / raw_image_path).resolve()
        if not raw_image_path or not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by {label_path.name} not found. imagePath={raw_image_path!r}"
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points: List[Tuple[float, float]] = []
        for shape in ann.get("shapes", []):
            if shape.get("label") != POINT_LABEL or shape.get("shape_type") != "point":
                continue
            raw_points = shape.get("points", [])
            if not raw_points:
                continue
            x, y = raw_points[0]
            points.append((float(x), float(y)))
        return image, points, ann

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if not self.training:
            return image

        out = image.astype(np.float32) / 255.0
        brightness_gain = float(self.augment_cfg.get("brightness_gain", 0.0))
        noise_std = float(self.augment_cfg.get("noise_std", 0.0))

        if brightness_gain > 0:
            alpha = 1.0 + np.random.uniform(-brightness_gain, brightness_gain)
            out = np.clip(out * alpha, 0.0, 1.0)

        if noise_std > 0:
            noise = np.random.normal(0.0, noise_std, out.shape).astype(np.float32)
            out = np.clip(out + noise, 0.0, 1.0)

        return (out * 255.0).astype(np.uint8)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | Dict | str]:
        sample_id = self.sample_ids[index]
        image, points, ann = self._load_labelme(sample_id)
        image = self._augment(image)

        orig_h, orig_w = image.shape[:2]
        resized, input_transform = resize_and_pad_image(image, (self.input_w, self.input_h))
        output_transform = build_resize_pad_transform(orig_w, orig_h, self.out_w, self.out_h)
        resized_f = normalize_rgb_image(resized, self.cfg)
        input_tensor = torch.from_numpy(resized_f.transpose(2, 0, 1)).float()

        heatmap = np.zeros((1, self.out_h, self.out_w), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects,), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects,), dtype=np.uint8)
        gt_points = np.zeros((self.max_objects, 2), dtype=np.float32)
        gt_point_mask = np.zeros((self.max_objects,), dtype=np.uint8)
        gaussian_radius = self.heatmap_diameter_out / 2.0

        for obj_idx, (x, y) in enumerate(points[: self.max_objects]):
            gt_points[obj_idx] = (x, y)
            gt_point_mask[obj_idx] = 1
            x, y = transform_point((x, y), output_transform)
            if x < 0 or y < 0 or x >= self.out_w or y >= self.out_h:
                continue

            ct = np.array([x, y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(heatmap[0], (int(ct_int[0]), int(ct_int[1])), self.heatmap_diameter_out)
            ind[obj_idx] = ct_int[1] * self.out_w + ct_int[0]
            reg[obj_idx] = ct - ct_int
            reg_mask[obj_idx] = 1

        meta = {
            "sample_id": sample_id,
            "orig_size": [orig_w, orig_h],
            "input_size": [self.input_w, self.input_h],
            "point_count": len(points),
            "input_transform": input_transform,
            "output_transform": output_transform,
            "image_width": ann.get("imageWidth", orig_w),
            "image_height": ann.get("imageHeight", orig_h),
            "heatmap_diameter_out": self.heatmap_diameter_out,
            "gaussian_radius": gaussian_radius,
            "gaussian_radius_source": "fixed",
        }

        return {
            "image": input_tensor,
            "heatmap": torch.from_numpy(heatmap),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "gt_points": torch.from_numpy(gt_points),
            "gt_point_mask": torch.from_numpy(gt_point_mask),
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.int64),
            "sample_id": sample_id,
            "meta": meta,
        }
