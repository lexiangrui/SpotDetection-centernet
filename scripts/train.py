from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from centernet_spot.config import load_config
from centernet_spot.data import SpotDataset
from centernet_spot.decode import decode_single_prediction
from centernet_spot.evaluation import evaluate_threshold_sweep, select_best_threshold_metrics
from centernet_spot.losses import get_heatmap_loss, reg_l1_loss
from centernet_spot.model import SpotCenterNet
from centernet_spot.split import discover_labeled_ids, make_train_val_split, write_split_file
from centernet_spot.transforms import build_resize_pad_transform
from centernet_spot.utils import ensure_dir, get_device, save_json, set_seed
from centernet_spot.visualization import (
    add_panel_title,
    heatmap_to_gray,
    save_loss_curve,
    tensor_to_bgr,
)

ROOT = Path(__file__).resolve().parents[1]
VIS_INTERVAL = 20
DEFAULT_EVAL_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def refresh_splits(cfg: dict) -> dict[str, int]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    label_dir = root / data_cfg["label_dir"]
    split_dir = ensure_dir(root / data_cfg["split_dir"])

    sample_ids = discover_labeled_ids(label_dir)
    train_ids, val_ids = make_train_val_split(
        sample_ids,
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(cfg["seed"]),
    )

    write_split_file(split_dir / data_cfg["train_split"], train_ids)
    write_split_file(split_dir / data_cfg["val_split"], val_ids)

    stats = {
        "labeled_samples": len(sample_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
    }
    print(json.dumps({"splits_refreshed": stats}, ensure_ascii=False))
    return stats


def build_loader(cfg: dict, split_name: str, training: bool) -> DataLoader:
    dataset = SpotDataset(cfg, split_name=split_name, training=training)
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=training,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict,
    heatmap_loss_fn: Callable,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_hm_loss = 0.0
    total_reg_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        gt_heatmap = batch["heatmap"].to(device)
        gt_reg = batch["reg"].to(device)
        gt_ind = batch["ind"].to(device)
        gt_mask = batch["reg_mask"].to(device)

        outputs = model(images)
        hm_loss = heatmap_loss_fn(outputs["heatmap"], gt_heatmap)
        reg_loss = reg_l1_loss(outputs["reg"], gt_reg, gt_ind, gt_mask)
        loss = (
            hm_loss * float(cfg["train"]["heatmap_loss_weight"])
            + reg_loss * float(cfg["train"]["offset_loss_weight"])
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_hm_loss += float(hm_loss.item())
        total_reg_loss += float(reg_loss.item())

        if training and step % int(cfg["train"]["log_interval"]) == 0:
            print(json.dumps({
                "step": step,
                "loss": round(loss.item(), 6),
                "hm_loss": round(hm_loss.item(), 6),
                "reg_loss": round(reg_loss.item(), 6),
            }, ensure_ascii=False))

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "hm_loss": total_hm_loss / n,
        "reg_loss": total_reg_loss / n,
    }


def resolve_eval_cfg(cfg: dict) -> dict[str, float | int | list[float]]:
    infer_cfg = cfg.get("infer", {})
    eval_cfg = cfg.get("eval", {})

    raw_thresholds = eval_cfg.get("score_thresholds", DEFAULT_EVAL_THRESHOLDS)
    score_thresholds = sorted({round(float(thr), 6) for thr in raw_thresholds})
    if not score_thresholds:
        score_thresholds = [float(infer_cfg.get("score_threshold", 0.6))]

    return {
        "score_thresholds": score_thresholds,
        "topk": int(eval_cfg.get("topk", infer_cfg.get("topk", 256))),
        "nms_kernel": int(eval_cfg.get("nms_kernel", infer_cfg.get("nms_kernel", 5))),
        "match_radius_scale": float(eval_cfg.get("match_radius_scale", 0.3)),
        "min_match_radius": float(eval_cfg.get("min_match_radius", 3.0)),
    }


def round_metrics(metrics: dict[str, float | int | None]) -> dict[str, float | int | None]:
    rounded: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict[str, dict | list[dict]]:
    eval_cfg = resolve_eval_cfg(cfg)
    score_thresholds = list(eval_cfg["score_thresholds"])
    min_score_threshold = float(score_thresholds[0])

    predictions_by_image: list[list[dict]] = []
    gt_points_by_image: list[np.ndarray] = []
    match_radii: list[float] = []

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            heatmaps = outputs["heatmap"]
            regs = outputs["reg"]
            out_h, out_w = heatmaps.shape[-2:]

            orig_sizes = batch["orig_size"]
            gt_points = batch["gt_points"]
            gt_point_mask = batch["gt_point_mask"]
            spot_diameter_orig = batch["spot_diameter_orig"]

            for sample_idx in range(images.shape[0]):
                orig_w = int(orig_sizes[sample_idx, 0].item())
                orig_h = int(orig_sizes[sample_idx, 1].item())
                output_transform = build_resize_pad_transform(orig_w, orig_h, out_w, out_h)
                detections = decode_single_prediction(
                    heatmap=heatmaps[sample_idx],
                    reg=regs[sample_idx],
                    transform=output_transform,
                    topk=int(eval_cfg["topk"]),
                    score_threshold=min_score_threshold,
                    nms_kernel=int(eval_cfg["nms_kernel"]),
                )
                valid_gt_points = gt_points[sample_idx][gt_point_mask[sample_idx].bool()].cpu().numpy()
                match_radius = max(
                    float(spot_diameter_orig[sample_idx].item()) * float(eval_cfg["match_radius_scale"]),
                    float(eval_cfg["min_match_radius"]),
                )

                predictions_by_image.append(detections)
                gt_points_by_image.append(valid_gt_points.astype(np.float32))
                match_radii.append(match_radius)

    if model_was_training:
        model.train()

    threshold_metrics = evaluate_threshold_sweep(
        predictions_by_image=predictions_by_image,
        gt_points_by_image=gt_points_by_image,
        match_radii=match_radii,
        score_thresholds=score_thresholds,
    )
    best_metrics = select_best_threshold_metrics(threshold_metrics)
    return {
        "best": best_metrics,
        "threshold_metrics": threshold_metrics,
    }


def _location_for_compare(metrics: dict[str, float | int | None] | None) -> float:
    if not metrics or metrics.get("mean_loc_error") is None:
        return float("inf")
    return float(metrics["mean_loc_error"])


def is_better_eval_candidate(
    current_eval: dict[str, float | int | None],
    current_val_loss: float,
    best_eval: dict[str, float | int | None] | None,
    best_val_loss: float,
    min_delta: float,
) -> bool:
    if best_eval is None:
        return True

    current_f1 = float(current_eval["f1"])
    best_f1 = float(best_eval["f1"])
    if current_f1 > best_f1 + min_delta:
        return True
    if current_f1 < best_f1 - min_delta:
        return False

    current_loc = _location_for_compare(current_eval)
    best_loc = _location_for_compare(best_eval)
    if current_loc < best_loc - 1e-8:
        return True
    if current_loc > best_loc + 1e-8:
        return False

    return current_val_loss < best_val_loss


def save_epoch_visualization(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    save_dir: Path,
    epoch: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    images = batch["image"][:1].to(device)
    gt_heatmap = batch["heatmap"][0]
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    if model_was_training:
        model.train()

    input_bgr = tensor_to_bgr(images[0], cfg)
    height, width = input_bgr.shape[:2]
    gt_gray = heatmap_to_gray(gt_heatmap, (width, height))
    pred_gray = heatmap_to_gray(outputs["heatmap"][0], (width, height))

    panels = [
        add_panel_title(input_bgr, "Input Image"),
        add_panel_title(gt_gray, "Ground Truth"),
        add_panel_title(pred_gray, "Prediction"),
    ]
    gap = 20
    canvas_h = max(panel.shape[0] for panel in panels)
    canvas_w = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    canvas = np.full((canvas_h, canvas_w, 3), 235, dtype=np.uint8)

    x = 0
    for panel in panels:
        panel_h, panel_w = panel.shape[:2]
        y = (canvas_h - panel_h) // 2
        canvas[y : y + panel_h, x : x + panel_w] = panel
        x += panel_w + gap

    vis_dir = ensure_dir(save_dir / "train_vis")
    out_path = vis_dir / f"epoch_{epoch:03d}.jpg"
    import cv2
    cv2.imwrite(str(out_path), canvas)
    print(f"saved visualization: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CenterNet spot detector.")
    parser.add_argument("--config", type=str, default="configs/spot_centernet.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.save_dir is not None:
        cfg["train"]["save_dir"] = args.save_dir
    set_seed(int(cfg["seed"]))
    device = get_device()
    save_dir = ensure_dir(cfg["train"]["save_dir"])
    refresh_splits(cfg)

    heatmap_loss_type = cfg["train"].get("heatmap_loss_type", "mse")
    heatmap_loss_fn = get_heatmap_loss(heatmap_loss_type)
    print(f"Using heatmap loss: {heatmap_loss_type}")

    train_loader = build_loader(cfg, split_name="train", training=True)
    val_loader = build_loader(cfg, split_name="val", training=False)

    model = SpotCenterNet(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler_patience = int(cfg["train"].get("scheduler_patience", 4))
    scheduler_factor = float(cfg["train"].get("scheduler_factor", 0.5))
    scheduler = None
    if scheduler_patience >= 0 and 0.0 < scheduler_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    best_eval: dict[str, float | int | None] | None = None
    best_val = float("inf")
    best_eval_val_loss = float("inf")
    best_epoch = 0
    best_loss_epoch = 0
    selection_min_delta = float(cfg["train"].get("selection_min_delta", 1e-4))
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 12))
    epochs_without_improvement = 0
    history: list[dict] = []
    stop_reason = "completed"

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, heatmap_loss_fn)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, cfg=cfg, heatmap_loss_fn=heatmap_loss_fn)
        eval_result = evaluate_model(model, val_loader, device, cfg)
        eval_metrics = eval_result["best"]

        record = {
            "epoch": epoch,
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "train": {k: round(v, 6) for k, v in train_metrics.items()},
            "val": {k: round(v, 6) for k, v in val_metrics.items()},
            "eval": round_metrics(eval_metrics),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        last_payload = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "eval": eval_metrics,
            "threshold_metrics": eval_result["threshold_metrics"],
        }
        torch.save(last_payload, save_dir / "last.pt")

        if is_better_eval_candidate(
            current_eval=eval_metrics,
            current_val_loss=float(val_metrics["loss"]),
            best_eval=best_eval,
            best_val_loss=best_eval_val_loss,
            min_delta=selection_min_delta,
        ):
            best_eval = dict(eval_metrics)
            best_eval_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(last_payload, save_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_loss_epoch = epoch
            torch.save(last_payload, save_dir / "best_loss.pt")

        save_json(save_dir / "metrics.json", history)

        if epoch % VIS_INTERVAL == 0:
            save_epoch_visualization(model, val_loader, device, cfg, save_dir, epoch)

        if scheduler is not None:
            scheduler.step(float(eval_metrics["f1"]))

        summary = {
            "best_epoch": best_epoch,
            "best_eval_val_loss": round(best_eval_val_loss, 6) if best_epoch else None,
            "best_eval": round_metrics(best_eval) if best_eval is not None else None,
            "best_loss_epoch": best_loss_epoch if best_loss_epoch else None,
            "best_loss": round(best_val, 6) if best_loss_epoch else None,
            "epochs_ran": epoch,
            "early_stop_triggered": False,
            "early_stop_patience": early_stop_patience,
            "selection_min_delta": selection_min_delta,
            "scheduler_patience": scheduler_patience,
            "scheduler_factor": scheduler_factor,
            "stop_reason": stop_reason,
        }
        save_json(save_dir / "summary.json", summary)

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stop_reason = "early_stop"
            summary.update({
                "early_stop_triggered": True,
                "stop_reason": stop_reason,
            })
            save_json(save_dir / "summary.json", summary)
            print(
                f"early stopping at epoch {epoch}: no F1 improvement for {epochs_without_improvement} epochs"
            )
            break

    save_loss_curve(history, save_dir, project_root=ROOT)
    final_summary = {
        "best_epoch": best_epoch,
        "best_eval_val_loss": round(best_eval_val_loss, 6) if best_epoch else None,
        "best_eval": round_metrics(best_eval) if best_eval is not None else None,
        "best_loss_epoch": best_loss_epoch if best_loss_epoch else None,
        "best_loss": round(best_val, 6) if best_loss_epoch else None,
        "epochs_ran": len(history),
        "early_stop_triggered": stop_reason == "early_stop",
        "early_stop_patience": early_stop_patience,
        "selection_min_delta": selection_min_delta,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "stop_reason": stop_reason,
    }
    save_json(save_dir / "summary.json", final_summary)
    if best_eval is not None:
        print(
            "training finished, "
            f"best epoch={best_epoch}, "
            f"best f1={float(best_eval['f1']):.6f}, "
            f"best threshold={float(best_eval['score_threshold']):.3f}, "
            f"best loss={best_val:.6f}, device={device}"
        )
    else:
        print(f"training finished, best val loss={best_val:.6f}, device={device}")


if __name__ == "__main__":
    main()
