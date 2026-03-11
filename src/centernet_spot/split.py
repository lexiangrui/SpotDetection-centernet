from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple


def discover_labeled_ids(label_dir: str | Path) -> List[str]:
    label_dir = Path(label_dir)
    return sorted(p.stem for p in label_dir.glob("*.json"))


def make_train_val_split(
    sample_ids: Sequence[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not sample_ids:
        return [], []

    items = list(sample_ids)
    rng = random.Random(seed)
    rng.shuffle(items)

    val_count = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    val_ids = sorted(items[:val_count])
    train_ids = sorted(items[val_count:]) if val_count < len(items) else []

    if not train_ids and val_ids:
        train_ids = [val_ids.pop()]

    return train_ids, val_ids


def write_split_file(path: str | Path, sample_ids: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")


def read_split_file(path: str | Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
