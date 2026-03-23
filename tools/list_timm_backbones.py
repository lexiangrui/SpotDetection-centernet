from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "timm_backbone_report.json"


def classify_family(name: str) -> str:
    lowered = name.lower()
    if "mobilenet" in lowered or "ghostnet" in lowered or "mnasnet" in lowered or "fbnet" in lowered:
        return "lightweight_cnn"
    if "efficientnet" in lowered:
        return "efficientnet"
    if "regnet" in lowered:
        return "regnet"
    if "resnet" in lowered or "resnext" in lowered or "resnest" in lowered or "ecaresnet" in lowered:
        return "resnet_family"
    if "convnext" in lowered:
        return "convnext"
    if "swin" in lowered or "maxvit" in lowered or "coatnet" in lowered:
        return "hierarchical_transformer"
    if "vit" in lowered or "deit" in lowered or "beit" in lowered or "eva" in lowered:
        return "plain_transformer"
    return "other"


def looks_recommended(family: str, reductions: list[int]) -> bool:
    if family in {"lightweight_cnn", "efficientnet", "regnet", "resnet_family", "convnext"}:
        return reductions == [4, 8, 16, 32] or reductions == [2, 4, 8, 16, 32] or reductions[-4:] == [4, 8, 16, 32]
    return False


def inspect_model(model_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_name": model_name,
        "family": classify_family(model_name),
        "usable": False,
    }
    try:
        model = timm.create_model(model_name, pretrained=False, features_only=True)
    except Exception as exc:  # pragma: no cover - runtime probing
        record["error"] = f"create_model_failed: {type(exc).__name__}: {exc}"
        return record

    feature_info = getattr(model, "feature_info", None)
    if feature_info is None:
        record["error"] = "missing_feature_info"
        return record

    try:
        channels = [int(v) for v in feature_info.channels()]
        reductions = [int(v) for v in feature_info.reduction()]
    except Exception as exc:  # pragma: no cover - runtime probing
        record["error"] = f"feature_info_failed: {type(exc).__name__}: {exc}"
        return record

    record["channels"] = channels
    record["reductions"] = reductions

    if not channels or not reductions or len(channels) != len(reductions):
        record["error"] = "invalid_feature_info"
        return record

    try:
        dummy = torch.randn(1, 3, 640, 640)
        outputs = model(dummy)
    except Exception as exc:  # pragma: no cover - runtime probing
        record["error"] = f"forward_failed: {type(exc).__name__}: {exc}"
        return record

    if isinstance(outputs, tuple):
        outputs = list(outputs)
    if not isinstance(outputs, list) or not outputs:
        record["error"] = f"invalid_forward_output: {type(outputs).__name__}"
        return record

    shapes: list[list[int]] = []
    for feat in outputs:
        if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
            record["error"] = "non_4d_feature_found"
            return record
        shapes.append([int(v) for v in feat.shape])

    stride_values = [640 // shape[-1] for shape in shapes if shape[-1] > 0]
    if len(outputs) < 4:
        record["error"] = "too_few_feature_maps"
        record["feature_shapes"] = shapes
        return record

    record["usable"] = True
    record["feature_shapes"] = shapes
    record["stride_values"] = stride_values
    record["recommended"] = looks_recommended(record["family"], reductions)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe timm backbones usable by SpotCenterNet")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0, help="Only inspect first N timm models, 0 means all")
    args = parser.parse_args()

    model_names = sorted(timm.list_models(pretrained=False))
    if args.limit > 0:
        model_names = model_names[: args.limit]

    records = []
    for idx, model_name in enumerate(model_names, start=1):
        record = inspect_model(model_name)
        records.append(record)
        status = "OK" if record.get("usable") else "SKIP"
        print(f"[{idx}/{len(model_names)}] {status} {model_name}")

    usable = [r for r in records if r.get("usable")]
    recommended = [r for r in usable if r.get("recommended")]

    payload = {
        "总模型数": len(records),
        "可用模型数": len(usable),
        "推荐模型数": len(recommended),
        "可用模型": usable,
        "推荐模型": recommended,
        "全日志": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved report to: {output_path}")
    print(json.dumps({
        "总模型数": len(records),
        "可用模型数": len(usable),
        "推荐模型数": len(recommended),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
