from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from centernet_spot.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX model to RKNN for RK3576.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to static ONNX model.")
    parser.add_argument("--output", type=str, required=True, help="Output RKNN path.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spot_centernet_resnet18.yaml",
        help="Project config used to derive RKNN preprocessing parameters.",
    )
    parser.add_argument("--target-platform", type=str, default="rk3576")
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Calibration dataset txt file, required when --quantize is enabled.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_rknn_preprocess(cfg: dict) -> tuple[list[list[float]], list[list[float]]]:
    mean = [float(v) * 255.0 for v in cfg["data"]["normalize_mean"]]
    std = [float(v) * 255.0 for v in cfg["data"]["normalize_std"]]
    return [mean], [std]


def main() -> None:
    args = parse_args()

    if args.quantize and not args.dataset:
        raise ValueError("--quantize requires --dataset")

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise SystemExit(
            "rknn-toolkit2 is not installed in the current Python environment."
        ) from exc

    cfg = load_config(args.config)
    mean_values, std_values = get_rknn_preprocess(cfg)

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=args.verbose)

    ret = rknn.config(
        target_platform=args.target_platform,
        mean_values=mean_values,
        std_values=std_values,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    build_kwargs = {"do_quantization": bool(args.quantize)}
    if args.quantize:
        build_kwargs["dataset"] = args.dataset
    ret = rknn.build(**build_kwargs)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed with code {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")

    rknn.release()

    print(f"exported RKNN to {output_path}")
    print(f"target_platform={args.target_platform}")
    print(f"quantize={args.quantize}")
    print(f"mean_values={mean_values}")
    print(f"std_values={std_values}")


if __name__ == "__main__":
    main()
