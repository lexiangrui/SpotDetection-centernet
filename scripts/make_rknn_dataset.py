#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from centernet_spot.transforms import resize_and_pad_image

def preprocess(image: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, _ = resize_and_pad_image(image, (input_w, input_h))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare RKNN quantization dataset from photos')
    parser.add_argument('--photos', default='/media/psf/Downloads/光斑定位-centernet/photos')
    parser.add_argument('--out-dir', default='/media/psf/Downloads/光斑定位-centernet/outputs/spot_centernet_resnet18/rknn_dataset')
    parser.add_argument('--dataset-txt', default='/media/psf/Downloads/光斑定位-centernet/outputs/spot_centernet_resnet18/rknn_dataset.txt')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--input-width', type=int, default=512)
    parser.add_argument('--input-height', type=int, default=512)
    args = parser.parse_args()

    photos_dir = Path(args.photos)
    out_dir = Path(args.out_dir)
    dataset_txt = Path(args.dataset_txt)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in photos_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise RuntimeError(f'No images found under {photos_dir}')

    lines = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f'skip unreadable image: {image_path}')
            continue
        arr = preprocess(image, args.input_width, args.input_height)
        npy_path = out_dir / f'{image_path.stem}.npy'
        np.save(npy_path, arr)
        lines.append(str(npy_path))
        print(f'prepared: {npy_path}')

    dataset_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'dataset txt: {dataset_txt}')
    print(f'samples: {len(lines)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
