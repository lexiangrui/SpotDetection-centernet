from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from torch import nn

from centernet_spot.config import load_config
from centernet_spot.model import SpotCenterNet


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(images)
        return outputs["heatmap"], outputs["reg"]


def resolve_config(args_config: str | None, checkpoint: dict) -> dict:
    if args_config:
        return load_config(args_config)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    raise ValueError("Config not found. Pass --config or use a checkpoint containing a 'config' field.")


def resolve_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
        if isinstance(model_state, dict):
            return model_state
    if isinstance(checkpoint, dict) and checkpoint:
        first_value = next(iter(checkpoint.values()))
        if isinstance(first_value, torch.Tensor):
            return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected {'model': state_dict, ...} or a plain state_dict.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SpotCenterNet checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = resolve_config(args.config, checkpoint)
    state_dict = resolve_state_dict(checkpoint)

    model = SpotCenterNet(cfg)
    model.load_state_dict(state_dict)
    model.eval()

    input_h = int(cfg["data"]["input_height"])
    input_w = int(cfg["data"]["input_width"])
    dummy_input = torch.randn(args.batch_size, 3, input_h, input_w, dtype=torch.float32)

    export_model = OnnxExportWrapper(model)
    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "heatmap": {0: "batch"},
            "reg": {0: "batch"},
        }

    export_kwargs = {
        "input_names": ["images"],
        "output_names": ["heatmap", "reg"],
        "opset_version": args.opset,
        "dynamic_axes": dynamic_axes,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            **export_kwargs,
        )

    print(f"exported ONNX to {output_path}")
    print(f"input: images[{args.batch_size}, 3, {input_h}, {input_w}]")
    print("outputs: heatmap, reg")


if __name__ == "__main__":
    main()
