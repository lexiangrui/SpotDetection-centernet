#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rknn.api import RKNN

from centernet_spot.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for CenterNet spot model")
    parser.add_argument(
        "--onnx",
        default="/media/psf/Downloads/光斑定位-centernet/outputs/spot_centernet_resnet18/best_static.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output",
        default="/media/psf/Downloads/gstv/resouces/spot_centernet_resnet18.rknn",
        help="Path to output RKNN model",
    )
    parser.add_argument(
        "--target-platform",
        default="rk3576",
        help="RKNN target platform, e.g. rk3576/rk3588",
    )
    parser.add_argument(
        "--config",
        default="configs/spot_centernet_resnet18.yaml",
        help="Project config used to derive RKNN preprocessing parameters",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=640,
        help="Model input width",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=384,
        help="Model input height",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable quantization build",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset txt path used when --quantize is enabled",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    load_config(args.config)

    if not onnx_path.exists():
        print(f"[ERR] ONNX not found: {onnx_path}", file=sys.stderr)
        return 1

    if args.quantize and not args.dataset:
        print("[ERR] --quantize requires --dataset", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        print(f"[INFO] Config target_platform={args.target_platform}")
        ret = rknn.config(
            target_platform=args.target_platform,
            optimization_level=3,
            mean_values=[[0.0, 0.0, 0.0]],
            std_values=[[255.0, 255.0, 255.0]],
        )
        if ret != 0:
            print(f"[ERR] rknn.config failed: {ret}", file=sys.stderr)
            return ret

        print(f"[INFO] Loading ONNX: {onnx_path}")
        ret = rknn.load_onnx(
            model=str(onnx_path),
        )
        if ret != 0:
            print(f"[ERR] rknn.load_onnx failed: {ret}", file=sys.stderr)
            return ret

        print(f"[INFO] Building RKNN, quantize={args.quantize}")
        ret = rknn.build(
            do_quantization=args.quantize,
            dataset=args.dataset if args.quantize else None,
        )
        if ret != 0:
            print(f"[ERR] rknn.build failed: {ret}", file=sys.stderr)
            return ret

        print(f"[INFO] Exporting RKNN: {output_path}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            print(f"[ERR] rknn.export_rknn failed: {ret}", file=sys.stderr)
            return ret

        print(f"[OK] RKNN exported to: {output_path}")
        print("[INFO] mean_values=[[0.0, 0.0, 0.0]]")
        print("[INFO] std_values=[[255.0, 255.0, 255.0]]")
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    raise SystemExit(main())
