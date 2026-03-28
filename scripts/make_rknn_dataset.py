#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from centernet_spot.config import load_config
from centernet_spot.transforms import resize_and_pad_image

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
    parser.add_argument(
        "--layout",
        choices=("nchw", "nhwc"),
        default="nchw",
        help="Tensor layout written into each .npy sample. RKNN build usually expects the model input layout.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_w = int(args.input_width or cfg["data"]["input_width"])
    input_h = int(args.input_height or cfg["data"]["input_height"])

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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized, _ = resize_and_pad_image(image_rgb, (input_w, input_h))
        arr = image_resized.astype(np.uint8)
        if args.layout == "nchw":
            arr = np.expand_dims(arr.transpose(2, 0, 1), axis=0)
        else:
            arr = np.expand_dims(arr, axis=0)
        npy_path = out_dir / f"{image_path.stem}.npy"
        np.save(npy_path, arr)
        lines.append(str(npy_path))
        print(f"prepared: {npy_path}")

    dataset_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"dataset txt: {dataset_txt}")
    print(f"samples: {len(lines)}")
    print(f"layout: {args.layout}, tensor shape example: [1, 3, {input_h}, {input_w}]"
          if args.layout == "nchw"
          else f"layout: {args.layout}, tensor shape example: [1, {input_h}, {input_w}, 3]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
