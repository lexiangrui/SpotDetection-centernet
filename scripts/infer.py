from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from centernet_spot.config import load_config
from centernet_spot.decode import decode_predictions
from centernet_spot.model import SpotCenterNet, heatmap_probs_from_logits
from centernet_spot.preprocessing import preprocess_image
from centernet_spot.transforms import build_resize_pad_transform
from centernet_spot.utils import assign_spot_ids, ensure_dir, get_device, save_json
from centernet_spot.visualization import (
    draw_detections_cross,
    make_heatmap_vis,
    make_three_panel,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


def collect_media(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
        )
    return [input_path]


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
    tensor = preprocess_image(image, cfg).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        heatmap = heatmap_probs_from_logits(outputs["heatmap_logits"])
        output_transform = build_resize_pad_transform(
            orig_w, orig_h,
            heatmap.shape[-1],
            heatmap.shape[-2],
        )
        detections = decode_predictions(
            heatmap=heatmap,
            reg=outputs["reg"],
            transform=output_transform,
            topk=topk,
            score_threshold=float(score_threshold),
            nms_kernel=nms_kernel,
        )
    outputs["heatmap"] = heatmap
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
    panel_right = make_annotated_frame(image, detections)
    return make_three_panel(panel_left, panel_mid, panel_right)


def make_annotated_frame(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    return draw_detections_cross(image, detections)


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
        model=model, image=image, cfg=cfg, device=device,
        score_threshold=score_threshold, topk=topk, nms_kernel=nms_kernel,
    )
    detections = assign_spot_ids(detections, image.shape[0])

    stem = image_path.stem
    vis = make_visualization(image, outputs["heatmap"], detections, output_transform)
    cv2.imwrite(str(output_dir / f"{stem}_vis.jpg"), vis)
    save_json(output_dir / f"{stem}.json", {
        "type": "image",
        "image": str(image_path),
        "count": len(detections),
        "detections": detections,
    })
    print(f"{image_path.name}: {len(detections)} detections")


def create_video_writer(
    output_dir: Path,
    stem: str,
    suffix: str,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    candidates = [
        (output_dir / f"{stem}_{suffix}.mp4", "mp4v"),
        (output_dir / f"{stem}_{suffix}.avi", "MJPG"),
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
    writer_vis: cv2.VideoWriter | None = None
    output_video_path: Path | None = None
    writer_annotated: cv2.VideoWriter | None = None
    annotated_video_path: Path | None = None

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            outputs, detections, output_transform = infer_detections(
                model=model, image=frame, cfg=cfg, device=device,
                score_threshold=score_threshold, topk=topk, nms_kernel=nms_kernel,
            )
            detections = assign_spot_ids(detections, frame.shape[0])
            annotated = make_annotated_frame(frame, detections)
            vis = make_visualization(frame, outputs["heatmap"], detections, output_transform)

            if writer_vis is None or writer_annotated is None:
                vis_h, vis_w = vis.shape[:2]
                ann_h, ann_w = annotated.shape[:2]
                writer_vis, output_video_path = create_video_writer(
                    output_dir, video_path.stem, "vis", fps, (vis_w, vis_h)
                )
                writer_annotated, annotated_video_path = create_video_writer(
                    output_dir, video_path.stem, "annotated", fps, (ann_w, ann_h)
                )

            writer_vis.write(vis)
            writer_annotated.write(annotated)

            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = frame_index * 1000.0 / fps
            frame_results.append({
                "frame_index": frame_index,
                "timestamp_ms": round(timestamp_ms, 3),
                "count": len(detections),
                "detections": detections,
            })

            frame_index += 1
            if frame_index % 30 == 0:
                print(f"{video_path.name}: processed {frame_index} frames")
    finally:
        capture.release()
        if writer_vis is not None:
            writer_vis.release()
        if writer_annotated is not None:
            writer_annotated.release()

    if output_video_path is None or annotated_video_path is None:
        print(f"skip empty video: {video_path}")
        return

    save_json(output_dir / f"{video_path.stem}.json", {
        "type": "video",
        "video": str(video_path),
        "visualization_video": str(output_video_path),
        "annotated_video": str(annotated_video_path),
        "fps": fps,
        "frame_count": len(frame_results),
        "frames": frame_results,
    })
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
    model = SpotCenterNet(load_pretrained_backbone=False).to(device)
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
                image_path=media_path, output_dir=output_dir, model=model,
                cfg=cfg, device=device, score_threshold=float(score_threshold),
                topk=topk, nms_kernel=nms_kernel,
            )
        elif suffix in VIDEO_SUFFIXES:
            process_video(
                video_path=media_path, output_dir=output_dir, model=model,
                cfg=cfg, device=device, score_threshold=float(score_threshold),
                topk=topk, nms_kernel=nms_kernel,
            )
        else:
            print(f"skip unsupported file: {media_path}")


if __name__ == "__main__":
    main()
