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
from centernet_spot.transforms import build_resize_pad_transform, resize_and_pad_image
from centernet_spot.utils import ensure_dir, get_device, save_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


def collect_media(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(
            [
                p for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
            ]
        )
    return [input_path]


def preprocess(image: np.ndarray, cfg: dict) -> torch.Tensor:
    input_w = int(cfg["data"]["input_width"])
    input_h = int(cfg["data"]["input_height"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, _ = resize_and_pad_image(image, (input_w, input_h))
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()


def draw_crosshair(canvas: np.ndarray, cx: int, cy: int,
                    size: int = 6, color=(0, 255, 0), thickness: int = 1) -> None:
    """在 (cx, cy) 处画十字架标记。"""
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)


def draw_detections_cross(
    image: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
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
        cv2.putText(
            canvas,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def restore_map_to_original_size(
    heatmap: np.ndarray,
    output_transform: dict[str, float | int],
    target_h: int,
    target_w: int,
) -> np.ndarray:
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
    hm = heatmap_tensor.sigmoid().squeeze().detach().cpu().numpy()  # [H, W]
    hm = np.clip(hm, 0.0, 1.0)
    hm_resized = restore_map_to_original_size(hm, output_transform, target_h, target_w)
    hm_uint8 = (hm_resized * 255).astype(np.uint8)
    return np.repeat(hm_uint8[:, :, None], 3, axis=2)


def make_three_panel(original: np.ndarray, heatmap_gray: np.ndarray,
                     annotated: np.ndarray, gap: int = 4) -> np.ndarray:
    """拼接三栏图：左-原图  中-热力图  右-十字架标注图，中间用白色间隔。"""
    h, w = original.shape[:2]
    total_w = w * 3 + gap * 2
    canvas = np.full((h, total_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w + gap: 2 * w + gap] = heatmap_gray
    canvas[:, 2 * w + 2 * gap:] = annotated
    return canvas


def infer_detections(
    model: SpotCenterNet,
    image: np.ndarray,
    cfg: dict,
    device: torch.device,
    score_threshold: float,
    topk: int,
    nms_kernel: int,
 ) -> tuple[dict[str, torch.Tensor], list[dict], dict[str, float | int]]:
    orig_h, orig_w = image.shape[:2]
    tensor = preprocess(image, cfg)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        output_transform = build_resize_pad_transform(
            orig_w,
            orig_h,
            outputs["heatmap"].shape[-1],
            outputs["heatmap"].shape[-2],
        )
        detections = decode_predictions(
            heatmap=outputs["heatmap"],
            reg=outputs["reg"],
            transform=output_transform,
            topk=topk,
            score_threshold=float(score_threshold),
            nms_kernel=nms_kernel,
        )
    return outputs, detections, output_transform


def make_visualization(
    image: np.ndarray,
    heatmap: torch.Tensor,
    detections: list[dict],
    output_transform: dict[str, float | int],
) -> np.ndarray:
    orig_h, orig_w = image.shape[:2]
    panel_left = image.copy()
    panel_mid = make_heatmap_vis(heatmap, output_transform, orig_h, orig_w)
    panel_right = draw_detections_cross(image, detections)
    return make_three_panel(panel_left, panel_mid, panel_right)


def process_image(
    image_path: Path,
    output_dir: Path,
    model: SpotCenterNet,
    cfg: dict,
    device: torch.device,
    score_threshold: float,
    topk: int,
    nms_kernel: int,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"skip unreadable image: {image_path}")
        return

    outputs, detections, output_transform = infer_detections(
        model=model,
        image=image,
        cfg=cfg,
        device=device,
        score_threshold=score_threshold,
        topk=topk,
        nms_kernel=nms_kernel,
    )

    stem = image_path.stem
    vis = make_visualization(image, outputs["heatmap"], detections, output_transform)
    cv2.imwrite(str(output_dir / f"{stem}_vis.jpg"), vis)
    save_json(
        output_dir / f"{stem}.json",
        {
            "type": "image",
            "image": str(image_path),
            "count": len(detections),
            "detections": detections,
        },
    )
    print(f"{image_path.name}: {len(detections)} detections")


def create_video_writer(output_dir: Path, stem: str, fps: float, frame_size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    candidates = [
        (output_dir / f"{stem}_vis.mp4", "mp4v"),
        (output_dir / f"{stem}_vis.avi", "MJPG"),
    ]
    width, height = frame_size
    for output_path, codec in candidates:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, output_path
        writer.release()
    raise RuntimeError(f"Failed to create video writer for {stem}")


def process_video(
    video_path: Path,
    output_dir: Path,
    model: SpotCenterNet,
    cfg: dict,
    device: torch.device,
    score_threshold: float,
    topk: int,
    nms_kernel: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"skip unreadable video: {video_path}")
        return

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    frame_results: list[dict] = []
    writer: cv2.VideoWriter | None = None
    output_video_path: Path | None = None

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            outputs, detections, output_transform = infer_detections(
                model=model,
                image=frame,
                cfg=cfg,
                device=device,
                score_threshold=score_threshold,
                topk=topk,
                nms_kernel=nms_kernel,
            )
            vis = make_visualization(frame, outputs["heatmap"], detections, output_transform)

            if writer is None:
                vis_h, vis_w = vis.shape[:2]
                writer, output_video_path = create_video_writer(output_dir, video_path.stem, fps, (vis_w, vis_h))

            writer.write(vis)

            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = frame_index * 1000.0 / fps
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": round(timestamp_ms, 3),
                    "count": len(detections),
                    "detections": detections,
                }
            )

            frame_index += 1
            if frame_index % 30 == 0:
                print(f"{video_path.name}: processed {frame_index} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if output_video_path is None:
        print(f"skip empty video: {video_path}")
        return

    save_json(
        output_dir / f"{video_path.stem}.json",
        {
            "type": "video",
            "video": str(video_path),
            "visualization_video": str(output_video_path),
            "fps": fps,
            "frame_count": len(frame_results),
            "frames": frame_results,
        },
    )
    print(f"{video_path.name}: {len(frame_results)} frames processed")


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
    media_paths = collect_media(input_path)
    if not media_paths:
        raise RuntimeError(f"No supported images or videos found under {input_path}")

    score_threshold = args.score_threshold
    if score_threshold is None:
        score_threshold = float(cfg["infer"]["score_threshold"])

    topk = args.topk or int(cfg["infer"]["topk"])
    nms_kernel = int(cfg["infer"]["nms_kernel"])

    for media_path in media_paths:
        suffix = media_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            process_image(
                image_path=media_path,
                output_dir=output_dir,
                model=model,
                cfg=cfg,
                device=device,
                score_threshold=float(score_threshold),
                topk=topk,
                nms_kernel=nms_kernel,
            )
        elif suffix in VIDEO_SUFFIXES:
            process_video(
                video_path=media_path,
                output_dir=output_dir,
                model=model,
                cfg=cfg,
                device=device,
                score_threshold=float(score_threshold),
                topk=topk,
                nms_kernel=nms_kernel,
            )
        else:
            print(f"skip unsupported file: {media_path}")


if __name__ == "__main__":
    main()
