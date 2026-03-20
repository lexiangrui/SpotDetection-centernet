"""可视化工具：热力图渲染、检测标注、损失曲线绘制。"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .utils import denormalize_image, ensure_dir


def tensor_to_bgr(image_tensor: torch.Tensor, cfg: dict) -> np.ndarray:
    """将归一化后的 CHW tensor 转为 BGR uint8 图像。"""
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = denormalize_image(image, cfg)
    image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def heatmap_to_gray(heatmap_tensor: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """将 heatmap tensor 渲染为灰度三通道图像，resize 到指定 (w, h)。"""
    heatmap = heatmap_tensor.detach().cpu()
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
    heatmap_np = np.clip(heatmap.numpy(), 0.0, 1.0)
    heatmap_u8 = (heatmap_np * 255).astype(np.uint8)
    width, height = size
    heatmap_u8 = cv2.resize(heatmap_u8, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.repeat(heatmap_u8[:, :, None], 3, axis=2)


def restore_map_to_original_size(
    heatmap: np.ndarray,
    output_transform: dict[str, float | int],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """去除 padding 并 resize 到原始图片尺寸。"""
    left = int(output_transform["pad_left"])
    top = int(output_transform["pad_top"])
    resized_w = int(output_transform["resized_w"])
    resized_h = int(output_transform["resized_h"])
    cropped = heatmap[top : top + resized_h, left : left + resized_w]
    if cropped.size == 0:
        cropped = heatmap
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def make_heatmap_vis(
    heatmap_tensor: torch.Tensor,
    output_transform: dict[str, float | int],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """将模型输出的 heatmap 转换为灰度可视化图像。"""
    hm = heatmap_tensor.squeeze().detach().cpu().numpy()
    hm = np.clip(hm, 0.0, 1.0)
    hm_resized = restore_map_to_original_size(hm, output_transform, target_h, target_w)
    hm_uint8 = (hm_resized * 255).astype(np.uint8)
    return np.repeat(hm_uint8[:, :, None], 3, axis=2)


def draw_crosshair(canvas: np.ndarray, cx: int, cy: int,
                   size: int = 6, color=(0, 255, 0), thickness: int = 1) -> None:
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)


def draw_detections_cross(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """在图片上用十字架标记每个检测到的光斑。"""
    canvas = image.copy()
    min_side = min(image.shape[:2])
    marker_size = max(6, int(round(min_side * 0.012)))
    marker_thickness = max(1, int(round(marker_size / 4.5)))
    font_scale = max(0.36, marker_size / 15.0)
    text_thickness = max(1, marker_thickness)
    for det in detections:
        cx = int(round(det["x"]))
        cy = int(round(det["y"]))
        draw_crosshair(canvas, cx, cy, size=marker_size, color=(0, 255, 0), thickness=marker_thickness)
        label = f'{det["score"]:.2f}'
        label_pos = (cx + marker_size + 4, cy - max(marker_size // 2, 4))
        cv2.putText(canvas, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
    return canvas


def make_three_panel(original: np.ndarray, heatmap_gray: np.ndarray,
                     annotated: np.ndarray, gap: int = 4) -> np.ndarray:
    """拼接三栏图：左-原图  中-热力图  右-十字架标注图。"""
    h, w = original.shape[:2]
    total_w = w * 3 + gap * 2
    canvas = np.full((h, total_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w + gap: 2 * w + gap] = heatmap_gray
    canvas[:, 2 * w + 2 * gap:] = annotated
    return canvas


def add_panel_title(panel: np.ndarray, title: str) -> np.ndarray:
    title_h = 36
    canvas = np.full((panel.shape[0] + title_h, panel.shape[1], 3), 245, dtype=np.uint8)
    canvas[title_h:] = panel
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = max((panel.shape[1] - text_size[0]) // 2, 8)
    cv2.putText(canvas, title, (text_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def save_loss_curve(history: list[dict], save_dir: Path, project_root: Path | None = None) -> None:
    """使用 matplotlib 保存训练/验证损失曲线图。"""
    if not history:
        return

    epochs = [record["epoch"] for record in history]
    train_losses = [float(record["train"]["loss"]) for record in history]
    val_losses = [float(record["val"]["loss"]) for record in history]
    out_path = save_dir / "loss_curve.png"

    if project_root is not None:
        mpl_config_dir = ensure_dir(project_root / ".matplotlib")
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

    print(f"saved loss curve: {out_path}")
