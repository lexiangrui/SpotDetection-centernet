from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


POINT_LABEL = "spot"
POINT_SHAPE_TYPE = "point"
LEGACY_POINT_LABEL = "centroid"
LEGACY_BOX_LABEL = "spot"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

def resolve_image_path(split_dir: Path, ann: dict, sample_id: str) -> Path:
    raw_image_path = str(ann.get("imagePath", "")).strip()
    candidates: list[Path] = []
    if raw_image_path:
        candidates.append((split_dir / raw_image_path).resolve())
        candidates.append((split_dir / Path(raw_image_path).name).resolve())

    for suffix in IMAGE_SUFFIXES:
        candidates.append((split_dir / f"{sample_id}{suffix}").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found for {sample_id}: imagePath={raw_image_path!r}")


def extract_points(shapes: list[dict]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for shape in shapes:
        if shape.get("label") != LEGACY_POINT_LABEL or shape.get("shape_type") != POINT_SHAPE_TYPE:
            continue
        raw_points = shape.get("points", [])
        if not raw_points:
            continue
        x, y = raw_points[0]
        points.append((float(x), float(y)))
    if points:
        return points

    for shape in shapes:
        if shape.get("label") != POINT_LABEL or shape.get("shape_type") != POINT_SHAPE_TYPE:
            continue
        raw_points = shape.get("points", [])
        if not raw_points:
            continue
        x, y = raw_points[0]
        points.append((float(x), float(y)))
    if points:
        return points

    for shape in shapes:
        if shape.get("label") != LEGACY_BOX_LABEL or shape.get("shape_type") != "rectangle":
            continue
        raw_points = shape.get("points", [])
        if len(raw_points) < 2:
            continue
        xs = [float(pt[0]) for pt in raw_points]
        ys = [float(pt[1]) for pt in raw_points]
        points.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return points


def convert_annotation(json_path: Path) -> None:
    ann = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = resolve_image_path(json_path.parent, ann, json_path.stem)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image_h, image_w = image.shape[:2]
    points = extract_points(ann.get("shapes", []))
    shapes = [
        {
            "label": POINT_LABEL,
            "points": [[x, y]],
            "group_id": None,
            "description": "",
            "shape_type": POINT_SHAPE_TYPE,
            "flags": {},
            "mask": None,
        }
        for x, y in points
    ]

    converted = {
        "version": ann.get("version", "5.5.0"),
        "flags": ann.get("flags", {}),
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(ann.get("imageHeight", image_h)),
        "imageWidth": int(ann.get("imageWidth", image_w)),
    }
    json_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_split_dir(split_dir: Path) -> None:
    images_dir = split_dir / "images"
    if images_dir.exists():
        for path in list(images_dir.iterdir()):
            shutil.move(str(path), split_dir / path.name)
        images_dir.rmdir()

    labels_dir = split_dir / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)


def convert_dataset_root(dataset_root: Path) -> None:
    for split in ("train", "val"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        flatten_split_dir(split_dir)
        for json_path in sorted(split_dir.glob("*.json")):
            convert_annotation(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy dataset JSON to point-only format.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root, e.g. dataset1")
    args = parser.parse_args()

    convert_dataset_root(Path(args.dataset_root))
    print(f"converted dataset: {args.dataset_root}")


if __name__ == "__main__":
    main()
