"""
将 CenterNet Spot ONNX 模型转换为 RKNN 格式，支持 INT8 量化和 FP16 两种模式。

依赖:
    pip install rknn-toolkit2 --extra-index-url https://download.rockchip.com/rknn/rknn-toolkit2/latest/

用法:
    # INT8 量化（推荐，NPU 推理速度最快）:
    python scripts/export_rknn.py --checkpoint models/spot_centernet_resnet18_focal/best.pt \
        --output deploy/model/spot_centernet_int8.rknn --quantize int8

    # 不量化（FP16，浮点模型）:
    python scripts/export_rknn.py --checkpoint models/spot_centernet_resnet18_focal/best.pt \
        --output deploy/model/spot_centernet_fp16.rknn --quantize fp16

    # 复用已有 ONNX:
    python scripts/export_rknn.py --onnx models/spot_centernet_resnet18_focal/best.onnx \
        --output deploy/model/spot_centernet_int8.rknn
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

# ImageNet mean/std * 255 (for uint8 [0,255] input, used only in INT8 RKNN preprocessing)
IMAGENET_MEAN_U8 = [123.675, 116.28, 103.53]
IMAGENET_STD_U8 = [58.395, 57.12, 57.375]


# ---------------------------------------------------------------------------
# RKNN-target ONNX export (no normalization baked in)
# ---------------------------------------------------------------------------

class OnnxExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(images)
        return outputs["heatmap_logits"].sigmoid(), outputs["reg"]


def export_raw_onnx(checkpoint_path: Path, output_onnx: Path, cfg: dict,
                    batch_size: int = 1, opset: int = 17) -> None:
    """Export ONNX with raw RGB input expectation (no ImageNet normalization baked in).

    The exported ONNX still expects normalized RGB tensors, identical to PyTorch.
    For INT8 RKNN export, RKNN reproduces that normalization through mean/std
    preprocessing so the runtime path can feed uint8 RGB letterbox input directly.
    For pure floating-point RKNN export, deploy-side code must normalize on CPU
    and feed float input directly.
    """
    from centernet_spot.config import load_config
    from centernet_spot.model import SpotCenterNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if args_config := cfg.get("_arg_config"):
        resolved_cfg = load_config(args_config)
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        resolved_cfg = checkpoint["config"]
    else:
        raise ValueError("Config not found. Pass --config or use a checkpoint containing 'config'.")

    state_dict = checkpoint.get("model", checkpoint)
    if isinstance(state_dict, dict) and not any(isinstance(v, torch.Tensor) for v in state_dict.values()):
        first_val = next(iter(state_dict.values()))
        if isinstance(first_val, dict):
            state_dict = first_val

    model = SpotCenterNet(load_pretrained_backbone=False)
    model.load_state_dict(state_dict)
    model.eval()

    input_h = int(resolved_cfg["data"]["input_height"])
    input_w = int(resolved_cfg["data"]["input_width"])
    dummy_input = torch.randn(batch_size, 3, input_h, input_w, dtype=torch.float32)

    export_model = OnnxExportWrapper(model)

    with torch.no_grad():
        sample_outputs = model(dummy_input)
        torch.onnx.export(
            export_model,
            dummy_input,
            output_onnx,
            input_names=["images"],
            output_names=["heatmap", "reg"],
            opset_version=opset,
            do_constant_folding=True,
            *(dict(dynamo=False) if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}),
        )

    print(f"[onnx] exported -> {output_onnx}")
    print(f"[onnx] input:  images [{batch_size}, 3, {input_h}, {input_w}] float32")
    print(
        "[onnx] outputs: heatmap"
        f"{tuple(sample_outputs['heatmap_logits'].sigmoid().shape)} "
        f"reg{tuple(sample_outputs['reg'].shape)}"
    )


# ---------------------------------------------------------------------------
# Calibration dataset generation
# ---------------------------------------------------------------------------

def get_image_paths_for_calibration(
    split_file: Path,
    photos_dir: Path,
    max_images: int = 100,
) -> list[Path]:
    """Read image IDs from a split file and return full paths."""
    if not split_file.exists():
        print(f"[calib] split file not found: {split_file}, using all photos")
        photo_files = sorted(photos_dir.glob("*.jpg")) + sorted(photos_dir.glob("*.png"))
        return photo_files[:max_images]

    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    paths = []
    for img_id in ids:
        for suffix in (".jpg", ".jpeg", ".png"):
            p = photos_dir / f"{img_id}{suffix}"
            if p.exists():
                paths.append(p)
                break
    return paths[:max_images]


def build_calibration_dataset(
    image_paths: list[Path],
    cfg: dict,
    output_npy: Path,
) -> Path:
    """Preprocess images and save RKNN calibration data.

    For RKNN toolkit 2.3.x, using a dataset txt that lists per-sample .npy files
    is more reliable than passing one aggregated .npy tensor file.

    Returns the path to the generated dataset txt file.
    """
    from centernet_spot.transforms import resize_and_pad_image

    cache_dir = output_npy.parent / output_npy.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_txt = output_npy.with_suffix('.txt')

    input_h = int(cfg["data"]["input_height"])
    input_w = int(cfg["data"]["input_width"])
    lines = []
    kept = 0

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[calib] warning: cannot read {img_path}, skipping")
            continue
        # Match deploy path semantically for INT8 calibration: BGR -> RGB, letterbox,
        # uint8, no CPU normalization. RKNN toolkit 2.3.x still expects calibration .npy
        # samples in NCHW layout, so we store [1, 3, H, W].
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas_rgb, _ = resize_and_pad_image(image_rgb, (input_w, input_h))
        sample = np.expand_dims(canvas_rgb.transpose(2, 0, 1).astype(np.uint8), axis=0)  # [1, 3, H, W]
        sample_path = cache_dir / f"{img_path.stem}.npy"
        np.save(sample_path, sample)
        lines.append(str(sample_path))
        kept += 1
        if (i + 1) % 20 == 0:
            print(f"[calib] preprocessed {i + 1}/{len(image_paths)} images")

    if not lines:
        raise RuntimeError("No valid calibration images found.")

    dataset_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[calib] saved {kept} samples under {cache_dir}")
    print(f"[calib] dataset list -> {dataset_txt}")
    return dataset_txt


# ---------------------------------------------------------------------------
# RKNN conversion
# ---------------------------------------------------------------------------

def convert_to_rknn(
    onnx_path: Path,
    rknn_path: Path,
    calibration_dataset: Path,
    quantize: str,
    target_platform: str,
) -> None:
    """Convert ONNX to RKNN with quantization."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 not installed.", file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  pip install rknn-toolkit2 --extra-index-url https://download.rockchip.com/rknn/rknn-toolkit2/latest/", file=sys.stderr)
        sys.exit(1)

    rknn = RKNN(verbose=False)

    # rknn-toolkit2 compatibility:
    # - int8: use quantized_dtype='w8a8', do_quantization=True, and fuse mean/std
    # - fp16: do_quantization=False, no RKNN-side preprocessing, keep float16 input
    # - channel conversion uses quant_img_RGB2BGR instead of reorder_channel
    config_kwargs = dict(
        target_platform=target_platform,
        quant_img_RGB2BGR=False,
        optimization_level=3,
    )
    if quantize == "int8":
        config_kwargs["mean_values"] = [IMAGENET_MEAN_U8]
        config_kwargs["std_values"] = [IMAGENET_STD_U8]
        config_kwargs["quantized_dtype"] = "w8a8"
    else:
        config_kwargs["float_dtype"] = "float16"

    rknn.config(**config_kwargs)

    print(f"[rknn] loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"load_onnx failed with code {ret}")

    print(f"[rknn] building (quantize={quantize})...")
    ret = rknn.build(
        do_quantization=(quantize == "int8"),
        dataset=str(calibration_dataset) if quantize == "int8" else None,
    )
    if ret != 0:
        raise RuntimeError(f"build failed with code {ret}")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rknn] exporting to: {rknn_path}")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"export_rknn failed with code {ret}")

    rknn.release()
    print(f"[rknn] done: {rknn_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CenterNet Spot ONNX to RKNN.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint (.pt). Required if --onnx is not provided.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. Inferred from checkpoint if it contains 'config'.")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Skip ONNX export; use this existing ONNX file directly.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output RKNN path (.rknn).")
    parser.add_argument("--quantize", type=str, choices=["int8", "fp16"], default="int8",
                        help="导出模式: int8 (INT8 量化), fp16 (浮点模型)")
    parser.add_argument("--platform", type=str, default="rk3576",
                        help="目标平台，如 rk3576, rk3588, rk356x 等 (默认: rk3576).")
    parser.add_argument("--calib-split", type=str, default="splits/val.txt",
                        help="Split file for calibration images (default: splits/val.txt).")
    parser.add_argument("--photos-dir", type=str, default="photos",
                        help="Directory with source images (default: photos).")
    parser.add_argument("--calib-dir", type=str, default=".calib_cache",
                        help="Directory to cache calibration .npy (default: .calib_cache).")
    parser.add_argument("--calib-size", type=int, default=100,
                        help="Max calibration images (default: 100).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--reuse-calib", action="store_true",
                        help="Reuse cached calibration .npy if it exists.")
    args = parser.parse_args()

    if not args.checkpoint and not args.onnx:
        print("ERROR: provide either --checkpoint or --onnx", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    photos_dir = (project_root / args.photos_dir).resolve()
    calib_dir = (project_root / args.calib_dir).resolve()

    # Resolve config
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        cp = project_root / ckpt
        if cp.exists():
            checkpoint_for_cfg = cp
        else:
            checkpoint_for_cfg = ckpt
        raw = torch.load(checkpoint_for_cfg, map_location="cpu") if checkpoint_for_cfg.suffix == ".pt" else None

        if args.config:
            from centernet_spot.config import load_config
            cfg = load_config(args.config)
        elif isinstance(raw, dict) and isinstance(raw.get("config"), dict):
            cfg = raw["config"]
        else:
            raise ValueError("Config not found. Pass --config or use a checkpoint with 'config' field.")
    else:
        # Use val split for config lookup
        val_split = project_root / args.calib_split
        if val_split.exists():
            from centernet_spot.config import load_config
            cfg = load_config(args.config or "configs/spot_centernet.yaml")
        else:
            from centernet_spot.config import load_config
            cfg = load_config(args.config or "configs/spot_centernet.yaml")

    input_h = int(cfg["data"]["input_height"])
    input_w = int(cfg["data"]["input_width"])
    output_rknn = Path(args.output)
    calib_dir.mkdir(parents=True, exist_ok=True)

    # ONNX export
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            onnx_path = project_root / onnx_path
        print(f"[onnx] using existing: {onnx_path}")
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            ckpt_path = project_root / ckpt_path
        onnx_path = ckpt_path.with_suffix(".onnx")
        print(f"[onnx] exporting from: {ckpt_path}")

        cfg["_arg_config"] = args.config  # pass through for resolve_config
        export_raw_onnx(ckpt_path, onnx_path, cfg, batch_size=args.batch_size, opset=args.opset)

    # Calibration dataset (INT8 only)
    calib_dataset = None
    if args.quantize == "int8":
        calib_npy = calib_dir / f"calib_{input_h}x{input_w}.npy"
        calib_dataset = calib_npy.with_suffix('.txt')
        split_file = project_root / args.calib_split

        if args.reuse_calib and calib_dataset.exists():
            print(f"[calib] reuse cached: {calib_dataset}")
        else:
            print(f"[calib] building from: {split_file}")
            image_paths = get_image_paths_for_calibration(split_file, photos_dir, args.calib_size)
            if not image_paths:
                print("ERROR: no calibration images found", file=sys.stderr)
                sys.exit(1)
            print(f"[calib] using {len(image_paths)} images")
            calib_dataset = build_calibration_dataset(image_paths, cfg, calib_npy)
    else:
        print("[calib] skipped for fp16 export")

    # RKNN conversion
    print(f"[rknn] converting: {onnx_path} -> {output_rknn}")
    print(f"[rknn] quantize={args.quantize}  platform={args.platform}  input={input_h}x{input_w}")
    convert_to_rknn(
        onnx_path=onnx_path,
        rknn_path=output_rknn,
        calibration_dataset=calib_dataset,
        quantize=args.quantize,
        target_platform=args.platform,
    )

    print("\nDone!")
    print(f"  RKNN model : {output_rknn}")
    print(f"  Quantization: {args.quantize}")
    print(f"  Platform   : {args.platform}")
    if args.quantize == "int8":
        print(f"  Calibration: {calib_dataset} ({args.calib_size} images)")
        print("\n部署注意: INT8 模型输入 uint8 RGB [N,H,W,3]，mean/std 由 NPU 融合，")
        print("         detect() 无需额外归一化，直接传 NHWC uint8 canvas。")
    else:
        print("\n部署注意: FP16 模型不做量化，也不做 RKNN 侧 mean/std 预处理。")
        print("         部署端必须先在 CPU 上完成归一化，再按模型输入类型送入浮点数据。")
        print("         当前 deploy/src/spot_detector.cpp 已支持 FLOAT16/FLOAT32 输入路径。")


if __name__ == "__main__":
    main()
