#!/usr/bin/env python3
"""
scripts/prepare_dataset.py

The `prepare_data` stage of `dvc.yaml`: takes a flat, already-labeled
raw dataset (`raw_dir/images/*.jpg` + `raw_dir/labels/*.txt`, standard
YOLO-format label lines -- this script does NOT do annotation itself,
only organizes already-annotated data) and splits it deterministically
into `output_dir/images/{train,val}` + `output_dir/labels/{train,val}`,
writing an Ultralytics-format `data.yaml` pointing at the result.

Deliberately simple and dependency-free (stdlib only) -- this is a data
plumbing step, not a place to add opencv/numpy/etc. as dependencies just
to move files around.

Usage:
    python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/vehicles
    python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data/vehicles --val-fraction 0.15 --seed 7
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_labeled_pairs(raw_dir: Path) -> List[str]:
    """Return the stems of every image under `raw_dir/images` that has a
    matching `.txt` file under `raw_dir/labels` -- unlabeled images are
    skipped with nothing worse than a smaller dataset (not an error),
    since a partially-annotated raw dump is a completely normal state."""
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected '{images_dir}' to exist.")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected '{labels_dir}' to exist.")

    stems = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        if (labels_dir / f"{image_path.stem}.txt").exists():
            stems.append(image_path.stem)
    return stems


def split_stems(stems: List[str], val_fraction: float, seed: int) -> tuple:
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    num_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    return shuffled[num_val:], shuffled[:num_val]


def materialize_split(
    raw_dir: Path, output_dir: Path, split_name: str, stems: List[str]
) -> None:
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image_src = _find_image_file(raw_dir / "images", stem)
        label_src = raw_dir / "labels" / f"{stem}.txt"
        shutil.copy2(image_src, images_out / image_src.name)
        shutil.copy2(label_src, labels_out / label_src.name)


def _find_image_file(images_dir: Path, stem: str) -> Path:
    for ext in _IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image file found for stem '{stem}' under '{images_dir}'.")


def write_data_yaml(output_dir: Path, class_names: List[str]) -> Path:
    data_yaml_path = output_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    data_yaml_path.write_text(
        f"train: {output_dir.resolve() / 'images' / 'train'}\n"
        f"val: {output_dir.resolve() / 'images' / 'val'}\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n"
    )
    return data_yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["car", "motorcycle", "bus", "truck", "bicycle"],
        help="Ordered class names matching the label files' class ids.",
    )
    args = parser.parse_args()

    stems = find_labeled_pairs(args.raw_dir)
    if not stems:
        raise SystemExit(f"No labeled image/label pairs found under '{args.raw_dir}'.")

    train_stems, val_stems = split_stems(stems, args.val_fraction, args.seed)
    materialize_split(args.raw_dir, args.output_dir, "train", train_stems)
    materialize_split(args.raw_dir, args.output_dir, "val", val_stems)
    data_yaml_path = write_data_yaml(args.output_dir, args.class_names)

    print(
        f"Prepared {len(train_stems)} train / {len(val_stems)} val sample(s) "
        f"under '{args.output_dir}'. Wrote '{data_yaml_path}'."
    )


if __name__ == "__main__":
    main()
