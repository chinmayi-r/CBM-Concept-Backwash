#!/usr/bin/env python3
"""Visibility-aware relabeling of CUB attributes using CUB70 masks (prof note #1).

For every image that has CUB70 masks, flip each attribute label present(1)->0
when the body part that attribute describes is occluded (coarse mask area below
threshold). CUB labels are species-constant, so a 'tail pattern' attribute is
marked present even on photos where the tail is hidden; this corrects that.

Outputs (under --data-root):
  CUB_processed/<out-dir>/{train,val,test}.pkl   # relabeled copies
  cub70_relabel_diagnostics.parquet              # one row per (image, attribute):
        original_label, part, coarse_area_frac, coarse_visible, new_label, flipped

IMPORTANT CONSTRAINT: CUB70 masks exist only for the (test) images of the first
70 CUB classes. Records without masks are copied through unchanged. So a
visibility-aware *retrain* can only relabel masked images -- see notebook 02
Part B and curated/README #3 for how this bounds the prof-note-#3/#4 ablation.
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cub70_parts import attribute_to_part, COARSE_TO_CUB70


def coarse_visibility(vis: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Collapse 11 CUB70 parts -> coarse parts; return (image_name, coarse_part)
    area_frac and visible flag."""
    fine_to_coarse = {fine: coarse
                      for coarse, fines in COARSE_TO_CUB70.items() for fine in fines}
    v = vis.copy()
    v["coarse"] = v["part"].map(fine_to_coarse)
    v = v.dropna(subset=["coarse"])
    g = (v.groupby(["image_name", "coarse"])
           .agg(pixel_count=("pixel_count", "sum"),
                img_pixels=("img_pixels", "max"))
           .reset_index())
    g["area_frac"] = np.where(g.img_pixels > 0, g.pixel_count / g.img_pixels, 0.0)
    g["visible"] = g["area_frac"] >= threshold
    return g


def load_attribute_names(attr_names_file: str | Path = None, cub_root: str | Path = None) -> list[str]:
    """Load attribute names from file or from official CUB attributes.txt.

    If attr_names_file is provided, load from there. Otherwise try canonical CUB location.
    """
    if attr_names_file:
        path = Path(attr_names_file)
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]

    if cub_root is None:
        raise ValueError("Must provide either --attr-names or path to CUB root")

    cub_root = Path(cub_root)
    attrs_txt = cub_root / "attributes" / "attributes.txt"

    if not attrs_txt.exists():
        raise FileNotFoundError(
            f"No attributes.txt found at {attrs_txt}. "
            f"Expected from official CUB-200-2011 release. "
            f"Or provide --attr-names manually."
        )

    # Parse attributes.txt: one line per attribute, format: "id::name" or "id name"
    attrs_by_id = {}
    for line in attrs_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            try:
                attr_id = int(parts[0])
                name = parts[1].strip()
                attrs_by_id[attr_id] = name
            except ValueError:
                continue

    if not attrs_by_id:
        raise ValueError(f"Could not parse attribute names from {attrs_txt}")

    # Return names sorted by ID
    return [attrs_by_id[i] for i in sorted(attrs_by_id.keys())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--data-dir", default="CUB_processed/class_attr_data_10")
    ap.add_argument("--out-dir", default="CUB_processed/class_attr_data_10_relabeled")
    ap.add_argument("--attr-names", default=None,
                    help="text file, one attribute name per line (optional; defaults to "
                         "canonical CUB location from official release)")
    ap.add_argument("--vis-threshold", type=float, default=0.001)
    args = ap.parse_args()
    root = Path(args.data_root)
    src = root / args.data_dir
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load attribute names with fallback to canonical CUB location
    cub_root = root / "CUB_200_2011" if root else None
    attr_names = load_attribute_names(args.attr_names, cub_root)
    attr_part = [attribute_to_part(n) for n in attr_names]

    vis = pd.read_parquet(root / "cub70_visibility.parquet")
    cv = coarse_visibility(vis, args.vis_threshold)
    # lookup: (image_name, coarse_part) -> visible
    vis_lut = {(r.image_name, r.coarse): bool(r.visible) for r in cv.itertuples()}

    diag_rows = []
    for split in ("train", "val", "test"):
        f = src / f"{split}.pkl"
        if not f.exists():
            continue
        recs = pickle.loads(f.read_bytes())
        for r in recs:
            stem = Path(r["img_path"]).stem
            labels = list(r["attribute_label"])
            for j, (lab, part) in enumerate(zip(labels, attr_part)):
                if part is None:
                    continue
                key = (stem, part)
                if key not in vis_lut:        # no mask for this image -> leave as-is
                    continue
                visible = vis_lut[key]
                new = 0 if (lab == 1 and not visible) else lab
                diag_rows.append({
                    "split": split, "image": stem, "attr_idx": j,
                    "attr_name": attr_names[j], "part": part,
                    "original_label": lab, "coarse_visible": visible,
                    "new_label": new, "flipped": int(new != lab),
                })
                labels[j] = new
            r["attribute_label"] = labels
        with open(out / f"{split}.pkl", "wb") as fh:
            pickle.dump(recs, fh)
        print(f"  relabeled {split}: {len(recs)} records")

    diag = pd.DataFrame(diag_rows)
    diag.to_parquet(root / "cub70_relabel_diagnostics.parquet", index=False)
    flipped = diag["flipped"].sum()
    considered = len(diag)
    print(f"wrote relabel diagnostics: {considered} (image,attr) pairs had masks, "
          f"{flipped} flipped present->absent ({100*flipped/max(considered,1):.1f}%)")
    if considered:
        print(diag[diag.flipped == 1].groupby("part").size().to_string())


if __name__ == "__main__":
    main()
