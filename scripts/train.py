from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from centernet_spot.utils import ensure_dir, get_device, save_json, set_seed


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

    print(f"training finished, best val loss={best_val:.6f}, device={device}")


if __name__ == "__main__":
    main()
