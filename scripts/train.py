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
from centernet_spot.losses import get_heatmap_loss, reg_l1_loss
from centernet_spot.model import SpotCenterNet
from centernet_spot.split import discover_labeled_ids, make_train_val_split, write_split_file
from centernet_spot.utils import ensure_dir, get_device, print_preprocessing_summary, save_json, set_seed
from centernet_spot.visualization import (
    add_panel_title,
    heatmap_to_gray,
    save_loss_curve,
    tensor_to_bgr,
)

ROOT = Path(__file__).resolve().parents[1]
VIS_INTERVAL = 20


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
    print_preprocessing_summary(cfg, prefix="train_preprocessing")
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

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, heatmap_loss_fn)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, cfg=cfg, heatmap_loss_fn=heatmap_loss_fn)

        record = {
            "epoch": epoch,
            "train": {k: round(v, 6) for k, v in train_metrics.items()},
            "val": {k: round(v, 6) for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        last_payload = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
        }
        torch.save(last_payload, save_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(last_payload, save_dir / "best.pt")

        save_json(save_dir / "metrics.json", history)

        if epoch % VIS_INTERVAL == 0:
            save_epoch_visualization(model, val_loader, device, cfg, save_dir, epoch)

    save_loss_curve(history, save_dir, project_root=ROOT)
    print(f"training finished, best val loss={best_val:.6f}, device={device}")


if __name__ == "__main__":
    main()
