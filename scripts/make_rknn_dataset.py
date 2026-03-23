#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from centernet_spot.config import load_config
from centernet_spot.preprocessing import preprocess_image_numpy
from centernet_spot.utils import print_preprocessing_summary

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare RKNN quantization dataset from photos")
    parser.add_argument("--config", default=str(ROOT / "configs" / "spot_centernet.yaml"))
    parser.add_argument("--photos", default=str(ROOT / "photos"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "rknn_dataset"))
    parser.add_argument("--dataset-txt", default=str(ROOT / "outputs" / "rknn_dataset.txt"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print_preprocessing_summary(
        cfg,
        input_w=args.input_width,
        input_h=args.input_height,
        prefix="rknn_dataset_preprocessing",
    )

    photos_dir = Path(args.photos)
    out_dir = Path(args.out_dir)
    dataset_txt = Path(args.dataset_txt)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in photos_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise RuntimeError(f"No images found under {photos_dir}")

    lines = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue
        arr = preprocess_image_numpy(image, cfg, input_w=args.input_width, input_h=args.input_height)
        npy_path = out_dir / f"{image_path.stem}.npy"
        np.save(npy_path, arr)
        lines.append(str(npy_path))
        print(f"prepared: {npy_path}")

    dataset_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"dataset txt: {dataset_txt}")
    print(f"samples: {len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
