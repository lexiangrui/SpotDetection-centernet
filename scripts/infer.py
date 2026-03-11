from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from centernet_spot.config import load_config
from centernet_spot.decode import decode_predictions
from centernet_spot.model import SpotCenterNet
from centernet_spot.transforms import get_affine_transform
from centernet_spot.utils import ensure_dir, get_device, save_json


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return [input_path]


def preprocess(image: np.ndarray, cfg: dict) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    input_w = int(cfg["data"]["input_width"])
    input_h = int(cfg["data"]["input_height"])
    orig_h, orig_w = image.shape[:2]
    center = np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32)
    scale = np.array([orig_w, orig_h], dtype=np.float32)
    trans_input = get_affine_transform(center, scale, 0, (input_w, input_h))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.warpAffine(image, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array(cfg["data"]["normalize_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg["data"]["normalize_std"], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return image, center, scale


def draw_crosshair(canvas: np.ndarray, cx: int, cy: int,
                    size: int = 6, color=(0, 255, 0), thickness: int = 1) -> None:
    """在 (cx, cy) 处画十字架标记。"""
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)


def draw_detections_cross(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """在图片上用十字架标记每个检测到的光斑。"""
    canvas = image.copy()
    for det in detections:
        cx = int(round(det["x"]))
        cy = int(round(det["y"]))
        draw_crosshair(canvas, cx, cy, size=6, color=(0, 255, 0), thickness=1)
        cv2.putText(
            canvas,
            f'{det["score"]:.2f}',
            (cx + 8, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_heatmap_vis(heatmap_tensor: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    """将模型输出的 heatmap 转换为彩色可视化图像 (BGR)。"""
    hm = heatmap_tensor.sigmoid().squeeze().detach().cpu().numpy()  # [H, W]
    hm = np.clip(hm, 0.0, 1.0)
    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_resized = cv2.resize(hm_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    return hm_color


def make_three_panel(original: np.ndarray, heatmap_color: np.ndarray,
                     annotated: np.ndarray, gap: int = 4) -> np.ndarray:
    """拼接三栏图：左-原图  中-热力图  右-十字架标注图，中间用白色间隔。"""
    h, w = original.shape[:2]
    total_w = w * 3 + gap * 2
    canvas = np.full((h, total_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w + gap: 2 * w + gap] = heatmap_color
    canvas[:, 2 * w + 2 * gap:] = annotated
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CenterNet spot inference.")
    parser.add_argument("--config", type=str, default="configs/spot_centernet.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/infer")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    output_dir = ensure_dir(args.output)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SpotCenterNet(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_path = Path(args.input)
    image_paths = collect_images(input_path)
    if not image_paths:
        raise RuntimeError(f"No images found under {input_path}")

    score_threshold = args.score_threshold
    if score_threshold is None:
        score_threshold = float(cfg["infer"]["score_threshold"])

    topk = args.topk or int(cfg["infer"]["topk"])
    nms_kernel = int(cfg["infer"]["nms_kernel"])

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue

        tensor, center, scale = preprocess(image, cfg)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)
            detections = decode_predictions(
                heatmap=outputs["heatmap"],
                reg=outputs["reg"],
                center=center,
                scale=scale,
                output_width=outputs["heatmap"].shape[-1],
                output_height=outputs["heatmap"].shape[-2],
                topk=topk,
                score_threshold=float(score_threshold),
                nms_kernel=nms_kernel,
            )

        stem = image_path.stem
        orig_h, orig_w = image.shape[:2]

        # 左：原图
        panel_left = image.copy()

        # 中：预测热力图
        panel_mid = make_heatmap_vis(outputs["heatmap"], orig_h, orig_w)

        # 右：十字架标注图
        panel_right = draw_detections_cross(image, detections)

        # 拼接三栏
        vis = make_three_panel(panel_left, panel_mid, panel_right)
        cv2.imwrite(str(output_dir / f"{stem}_vis.jpg"), vis)
        save_json(
            output_dir / f"{stem}.json",
            {
                "image": str(image_path),
                "count": len(detections),
                "detections": detections,
            },
        )
        print(f"{image_path.name}: {len(detections)} detections")


if __name__ == "__main__":
    main()
