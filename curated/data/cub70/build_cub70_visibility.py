#!/usr/bin/env python3
"""Compute per-image, per-part mask area from the CUB70 segmentation masks.

Output: $CURATED_DATA/cub70_visibility.parquet with columns
    image_name, class_idx, part (one of CUB70_PARTS), pixel_count,
    img_pixels, area_frac, visible
`visible` = area_frac >= --vis-threshold. This is the ground-truth visibility
table for the z-vs-occlusion analysis (prof notes #2/#4) and the relabeling
(prof note #1).

The released archive has this verified layout:
  masks/AnnotationMasksPerclass/<class_id>/<image_stem>_<part>.png

Only visible/annotated parts have a file. A missing part file for an otherwise
annotated image is therefore recorded with pixel_count=0.
"""
from __future__ import annotations
import argparse
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image

from cub70_parts import CUB70_PARTS

def _read_gray(path: Path) -> np.ndarray:
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return np.array(Image.open(path).convert("L"))


def _archive_root(masks_root: Path) -> Path:
    nested = masks_root / "AnnotationMasksPerclass"
    return nested if nested.is_dir() else masks_root


def iter_per_part_masks(masks_root: Path):
    """Yield (image_name, class_idx, {part: pixel_count}, img_pixels)."""
    root = _archive_root(masks_root)
    suffix = re.compile(r"_(" + "|".join(map(re.escape, sorted(CUB70_PARTS, key=len, reverse=True))) + r")\.png$")
    for class_dir in sorted((p for p in root.iterdir() if p.is_dir()),
                            key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        try:
            class_idx = int(class_dir.name.split(".")[0]) - 1
        except ValueError:
            class_idx = -1
        grouped = {}
        for f in class_dir.glob("*.png"):
            m = suffix.search(f.name)
            if not m:
                continue
            part = m.group(1)
            stem = f.name[:m.start()]
            grouped.setdefault(stem, {})[part] = f
        for stem, files in sorted(grouped.items()):
            counts = {part: 0 for part in CUB70_PARTS}
            img_pixels = 0
            for part, f in files.items():
                mask = _read_gray(f)
                if mask is None:
                    raise ValueError(f"could not read mask: {f}")
                img_pixels = max(img_pixels, int(mask.size))
                counts[part] = int((mask > 0).sum())
            yield stem, class_idx, counts, img_pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--vis-threshold", type=float, default=0.001,
                    help="min mask area fraction to call a part 'visible'")
    args = ap.parse_args()
    root = Path(args.data_root)
    masks_root = root / "cub70" / "masks"
    if not masks_root.is_dir():
        raise FileNotFoundError(f"missing {masks_root}; run data/cub70/fetch_cub70_masks.sh")

    rows = []
    for image_name, class_idx, counts, img_pixels in iter_per_part_masks(masks_root):
        for part, px in counts.items():
            frac = (px / img_pixels) if img_pixels else 0.0
            rows.append({
                "image_name": image_name, "class_idx": class_idx,
                "part": part, "pixel_count": px, "img_pixels": img_pixels,
                "area_frac": frac, "visible": frac >= args.vis_threshold,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no masks parsed under {masks_root}; archive layout is unexpected")
    if df.class_idx.min() < 0 or df.class_idx.max() >= 70:
        raise RuntimeError("CUB70 class directories did not parse as IDs 1..70")
    out = root / "cub70_visibility.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows, "
          f"{df.image_name.nunique()} images, {df.part.nunique()} parts")
    print(df.groupby("part")["visible"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
