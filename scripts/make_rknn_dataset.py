#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ], dtype=np.float32)


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center: np.ndarray, scale: np.ndarray, rot: float, output_size):
    if not isinstance(scale, np.ndarray):
        scale = np.array(scale, dtype=np.float32)

    src_w, src_h = scale[0], scale[1]
    dst_w, dst_h = output_size
    rot_rad = np.pi * rot / 180.0
    src_dir = get_dir([0, -0.5 * src_h], rot_rad)
    dst_dir = np.array([0, -0.5 * dst_h], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def preprocess(image: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    orig_h, orig_w = image.shape[:2]
    center = np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32)
    scale = np.array([orig_w, orig_h], dtype=np.float32)
    trans_input = get_affine_transform(center, scale, 0, (input_w, input_h))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.warpAffine(image, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare RKNN quantization dataset from photos')
    parser.add_argument('--photos', default='/media/psf/Downloads/光斑定位-centernet/photos')
    parser.add_argument('--out-dir', default='/media/psf/Downloads/光斑定位-centernet/outputs/spot_centernet_resnet18/rknn_dataset')
    parser.add_argument('--dataset-txt', default='/media/psf/Downloads/光斑定位-centernet/outputs/spot_centernet_resnet18/rknn_dataset.txt')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--input-width', type=int, default=640)
    parser.add_argument('--input-height', type=int, default=384)
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
