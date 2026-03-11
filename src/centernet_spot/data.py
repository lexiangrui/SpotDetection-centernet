from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .split import discover_labeled_ids, make_train_val_split, read_split_file
from .transforms import affine_transform, get_affine_transform


def gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=max(diameter / 6.0, 1e-3))
    x, y = center
    height, width = heatmap.shape

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    if min(left, right, top, bottom) <= 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3**2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class SpotDataset(Dataset):
    def __init__(self, cfg: Dict, split_name: str, training: bool) -> None:
        self.cfg = cfg
        self.training = training

        data_cfg = cfg["data"]
        self.root = Path(data_cfg["root"])
        self.image_dir = self.root / data_cfg["image_dir"]
        self.label_dir = self.root / data_cfg["label_dir"]
        self.split_dir = self.root / data_cfg["split_dir"]
        self.class_name = data_cfg["class_name"]
        self.spot_size_label = data_cfg.get("spot_size_label", "spot_size")
        self.spot_size_shape_type = data_cfg.get("spot_size_shape_type", "line")
        self.input_w = int(data_cfg["input_width"])
        self.input_h = int(data_cfg["input_height"])
        self.down_ratio = int(data_cfg["down_ratio"])
        self.out_w = self.input_w // self.down_ratio
        self.out_h = self.input_h // self.down_ratio
        self.max_objects = int(data_cfg["max_objects"])
        self.point_box_width = float(data_cfg["point_box_width"])
        self.point_box_height = float(data_cfg["point_box_height"])
        self.gaussian_min_overlap = float(data_cfg["gaussian_min_overlap"])
        self.min_gaussian_radius = int(data_cfg["min_gaussian_radius"])
        self.mean = np.array(data_cfg["normalize_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(data_cfg["normalize_std"], dtype=np.float32).reshape(1, 1, 3)
        self.augment_cfg = data_cfg.get("train_augment", {})

        split_file = self.split_dir / data_cfg[f"{split_name}_split"]
        if split_file.exists():
            self.sample_ids = read_split_file(split_file)
        else:
            all_ids = discover_labeled_ids(self.label_dir)
            train_ids, val_ids = make_train_val_split(
                all_ids,
                val_ratio=float(data_cfg["val_ratio"]),
                seed=int(cfg["seed"]),
            )
            self.sample_ids = train_ids if split_name == "train" else val_ids

        if not self.sample_ids:
            raise RuntimeError(f"No samples found for split={split_name}.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _parse_size_shape_segment(
        self, shape: Dict
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if shape.get("label") != self.spot_size_label:
            return None
        if shape.get("shape_type") != self.spot_size_shape_type:
            return None

        pts = shape.get("points", [])
        if len(pts) < 2:
            return None

        (x1, y1), (x2, y2) = pts[:2]
        return (float(x1), float(y1)), (float(x2), float(y2))

    def _load_labelme(
        self, sample_id: str
    ) -> Tuple[
        np.ndarray,
        List[Tuple[float, float]],
        List[Tuple[Tuple[float, float], Tuple[float, float]]],
        Dict,
    ]:
        label_path = self.label_dir / f"{sample_id}.json"
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        image_path = (label_path.parent / ann["imagePath"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points: List[Tuple[float, float]] = []
        size_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for shape in ann.get("shapes", []):
            if shape.get("label") == self.class_name and shape.get("shape_type") == "point":
                pt = shape.get("points", [])
                if not pt:
                    continue
                x, y = pt[0]
                points.append((float(x), float(y)))
                continue

            segment = self._parse_size_shape_segment(shape)
            if segment is not None:
                size_segments.append(segment)

        return image, points, size_segments, ann

    def _compute_gaussian_radius(
        self,
        size_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        trans_output: np.ndarray,
    ) -> Tuple[int, Optional[float]]:
        if not size_segments:
            radius = gaussian_radius(
                (
                    self.point_box_height / self.down_ratio,
                    self.point_box_width / self.down_ratio,
                ),
                min_overlap=self.gaussian_min_overlap,
            )
            return max(self.min_gaussian_radius, int(radius)), None

        transformed_diameters: List[float] = []
        for p1, p2 in size_segments:
            tp1 = np.array(affine_transform(p1, trans_output), dtype=np.float32)
            tp2 = np.array(affine_transform(p2, trans_output), dtype=np.float32)
            transformed_diameters.append(float(np.hypot(*(tp2 - tp1))))

        diameter_out = float(np.mean(transformed_diameters))
        radius = gaussian_radius((diameter_out, diameter_out), min_overlap=self.gaussian_min_overlap)
        return max(self.min_gaussian_radius, int(radius)), diameter_out

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
        image, points, size_segments, ann = self._load_labelme(sample_id)
        image = self._augment(image)

        orig_h, orig_w = image.shape[:2]
        center = np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32)
        scale = np.array([orig_w, orig_h], dtype=np.float32)
        trans_input = get_affine_transform(center, scale, 0, (self.input_w, self.input_h))
        trans_output = get_affine_transform(center, scale, 0, (self.out_w, self.out_h))

        resized = cv2.warpAffine(image, trans_input, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        resized_f = resized.astype(np.float32) / 255.0
        resized_f = (resized_f - self.mean) / self.std
        input_tensor = torch.from_numpy(resized_f.transpose(2, 0, 1)).float()

        heatmap = np.zeros((1, self.out_h, self.out_w), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects,), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects,), dtype=np.uint8)

        radius, spot_diameter = self._compute_gaussian_radius(size_segments, trans_output)

        for obj_idx, (x, y) in enumerate(points[: self.max_objects]):
            x, y = affine_transform((x, y), trans_output)
            if x < 0 or y < 0 or x >= self.out_w or y >= self.out_h:
                continue

            ct = np.array([x, y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(heatmap[0], (int(ct_int[0]), int(ct_int[1])), radius)
            ind[obj_idx] = ct_int[1] * self.out_w + ct_int[0]
            reg[obj_idx] = ct - ct_int
            reg_mask[obj_idx] = 1

        meta = {
            "sample_id": sample_id,
            "orig_size": [orig_w, orig_h],
            "input_size": [self.input_w, self.input_h],
            "point_count": len(points),
            "spot_size_annotation_count": len(size_segments),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "image_width": ann.get("imageWidth", orig_w),
            "image_height": ann.get("imageHeight", orig_h),
            "spot_diameter_out": spot_diameter,
            "gaussian_radius": radius,
            "gaussian_radius_source": "annotation" if spot_diameter is not None else "fallback_config",
        }

        return {
            "image": input_tensor,
            "heatmap": torch.from_numpy(heatmap),
            "reg": torch.from_numpy(reg),
            "ind": torch.from_numpy(ind),
            "reg_mask": torch.from_numpy(reg_mask),
            "meta": meta,
        }
