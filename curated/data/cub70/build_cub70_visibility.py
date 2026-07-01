#!/usr/bin/env python3
"""Compute per-image, per-part mask area from the CUB70 segmentation masks.

Output: $CURATED_DATA/cub70_visibility.parquet with columns
    image_name, class_idx, part (one of CUB70_PARTS), pixel_count,
    img_pixels, area_frac, visible
`visible` = area_frac >= --vis-threshold. This is the ground-truth visibility
table for the z-vs-occlusion analysis (prof notes #2/#4) and the relabeling
(prof note #1).

INPUT ASSUMPTIONS (adapt in one place if your CUB70 export differs):
  $CURATED_DATA/cub70/
    masks/<class_dir>/<image_name>/<part>.png   # binary mask per part, nonzero = part
  -- or a single indexed PNG per image with one integer id per part; pass
     --indexed and edit PART_INDEX below.
"""
from __future__ import annotations
import argparse
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

# only used with --indexed (single-label-map PNG); verify against CUB70 docs
PART_INDEX = {name: i + 1 for i, name in enumerate(CUB70_PARTS)}


def _read_gray(path: Path) -> np.ndarray:
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return np.array(Image.open(path).convert("L"))


def iter_per_part_masks(masks_root: Path):
    """Yield (image_name, class_idx, {part: pixel_count}, img_pixels)."""
    for class_dir in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        # class dir names like "001.Black_footed_Albatross"
        try:
            class_idx = int(class_dir.name.split(".")[0]) - 1
        except ValueError:
            class_idx = -1
        for img_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            counts, img_pixels = {}, None
            for part in CUB70_PARTS:
                f = img_dir / f"{part}.png"
                if not f.exists():
                    counts[part] = 0
                    continue
                m = _read_gray(f)
                if img_pixels is None:
                    img_pixels = int(m.size)
                counts[part] = int((m > 0).sum())
            yield img_dir.name, class_idx, counts, (img_pixels or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--vis-threshold", type=float, default=0.001,
                    help="min mask area fraction to call a part 'visible'")
    args = ap.parse_args()
    root = Path(args.data_root)
    masks_root = root / "cub70" / "masks"

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
    out = root / "cub70_visibility.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} rows, "
          f"{df.image_name.nunique()} images, {df.part.nunique()} parts")
    print(df.groupby("part")["visible"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
