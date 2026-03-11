from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from centernet_spot.split import discover_labeled_ids, make_train_val_split, write_split_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val split files.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--label-dir", type=str, default="labels_raw")
    parser.add_argument("--split-dir", type=str, default="splits")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    label_dir = root / args.label_dir
    split_dir = root / args.split_dir

    sample_ids = discover_labeled_ids(label_dir)
    train_ids, val_ids = make_train_val_split(sample_ids, val_ratio=args.val_ratio, seed=args.seed)

    write_split_file(split_dir / "train.txt", train_ids)
    write_split_file(split_dir / "val.txt", val_ids)

    print(f"labeled samples: {len(sample_ids)}")
    print(f"train samples: {len(train_ids)}")
    print(f"val samples: {len(val_ids)}")


if __name__ == "__main__":
    main()
