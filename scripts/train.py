from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from centernet_spot.config import load_config
from centernet_spot.data import SpotDataset
from centernet_spot.losses import focal_loss, reg_l1_loss
from centernet_spot.model import SpotCenterNet
from centernet_spot.split import discover_labeled_ids, make_train_val_split, write_split_file
from centernet_spot.utils import ensure_dir, get_device, save_json, set_seed

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
        hm_loss = focal_loss(outputs["heatmap"], gt_heatmap)
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
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(loss.item(), 6),
                        "hm_loss": round(hm_loss.item(), 6),
                        "reg_loss": round(reg_loss.item(), 6),
                    },
                    ensure_ascii=False,
                )
            )

    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "hm_loss": total_hm_loss / n,
        "reg_loss": total_reg_loss / n,
    }


def tensor_to_bgr_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def heatmap_to_gray(heatmap_tensor: torch.Tensor, size: tuple[int, int], apply_sigmoid: bool) -> np.ndarray:
    heatmap = heatmap_tensor.detach().cpu()
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    if apply_sigmoid:
        heatmap = heatmap.sigmoid()
    heatmap_np = np.clip(heatmap.numpy(), 0.0, 1.0)
    heatmap_u8 = (heatmap_np * 255).astype(np.uint8)
    width, height = size
    heatmap_u8 = cv2.resize(heatmap_u8, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.repeat(heatmap_u8[:, :, None], 3, axis=2)


def add_panel_title(panel: np.ndarray, title: str) -> np.ndarray:
    title_h = 36
    canvas = np.full((panel.shape[0] + title_h, panel.shape[1], 3), 245, dtype=np.uint8)
    canvas[title_h:] = panel
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = max((panel.shape[1] - text_size[0]) // 2, 8)
    text_y = 24
    cv2.putText(canvas, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


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

    input_bgr = tensor_to_bgr_image(images[0])
    height, width = input_bgr.shape[:2]
    gt_gray = heatmap_to_gray(gt_heatmap, (width, height), apply_sigmoid=False)
    pred_gray = heatmap_to_gray(outputs["heatmap"][0], (width, height), apply_sigmoid=True)

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
    cv2.imwrite(str(out_path), canvas)
    print(f"saved visualization: {out_path}")


def save_loss_curve(history: list[dict], save_dir: Path) -> None:
    if not history:
        return

    epochs = [record["epoch"] for record in history]
    train_losses = [float(record["train"]["loss"]) for record in history]
    val_losses = [float(record["val"]["loss"]) for record in history]
    out_path = save_dir / "loss_curve.png"

    try:
        mpl_config_dir = ensure_dir(ROOT / ".matplotlib")
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
        plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception:
        width, height = 1000, 520
        margin_left, margin_right = 90, 40
        margin_top, margin_bottom = 60, 70
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        plot_x0 = margin_left
        plot_y0 = margin_top
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        plot_x1 = plot_x0 + plot_w
        plot_y1 = plot_y0 + plot_h

        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (60, 60, 60), 1)
        cv2.putText(canvas, "Training and Validation Loss", (290, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Epoch", (width // 2 - 30, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Loss", (18, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

        min_loss = min(train_losses + val_losses)
        max_loss = max(train_losses + val_losses)
        if max_loss <= min_loss:
            max_loss = min_loss + 1.0

        def to_canvas_xy(epoch_value: int, loss_value: float) -> tuple[int, int]:
            if len(epochs) == 1:
                x = plot_x0
            else:
                x = plot_x0 + int((epoch_value - epochs[0]) / (epochs[-1] - epochs[0]) * plot_w)
            y = plot_y1 - int((loss_value - min_loss) / (max_loss - min_loss) * plot_h)
            return x, y

        for tick in range(5):
            y = plot_y0 + int(tick / 4 * plot_h)
            value = max_loss - (max_loss - min_loss) * tick / 4
            cv2.line(canvas, (plot_x0, y), (plot_x1, y), (220, 220, 220), 1)
            cv2.putText(canvas, f"{value:.2f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

        if len(epochs) > 1:
            for tick in range(min(6, len(epochs))):
                idx = round(tick * (len(epochs) - 1) / max(min(5, len(epochs) - 1), 1))
                x, _ = to_canvas_xy(epochs[idx], min_loss)
                cv2.line(canvas, (x, plot_y0), (x, plot_y1), (230, 230, 230), 1)
                cv2.putText(canvas, str(epochs[idx]), (x - 10, plot_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

        train_pts = np.array([to_canvas_xy(e, l) for e, l in zip(epochs, train_losses)], dtype=np.int32).reshape(-1, 1, 2)
        val_pts = np.array([to_canvas_xy(e, l) for e, l in zip(epochs, val_losses)], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [train_pts], False, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [val_pts], False, (0, 140, 255), 2, cv2.LINE_AA)

        legend_x = width - 190
        legend_y = 70
        cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 150, legend_y + 60), (210, 210, 210), -1)
        cv2.rectangle(canvas, (legend_x, legend_y), (legend_x + 150, legend_y + 60), (180, 180, 180), 1)
        cv2.line(canvas, (legend_x + 12, legend_y + 20), (legend_x + 42, legend_y + 20), (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Train Loss", (legend_x + 50, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(canvas, (legend_x + 12, legend_y + 45), (legend_x + 42, legend_y + 45), (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Val Loss", (legend_x + 50, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_path), canvas)

    print(f"saved loss curve: {out_path}")


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

    train_loader = build_loader(cfg, split_name="train", training=True)
    val_loader = build_loader(cfg, split_name="val", training=False)

    model = SpotCenterNet(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    best_val = float("inf")
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, cfg=cfg)

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

    save_loss_curve(history, save_dir)
    print(f"training finished, best val loss={best_val:.6f}, device={device}")


if __name__ == "__main__":
    main()
