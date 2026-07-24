#!/usr/bin/env python3
"""Visibility-aware relabeling of CUB attributes using CUB70 masks (prof note #1).

For every image that has CUB70 masks, flip each attribute label present(1)->0
when the body part that attribute describes is occluded (coarse mask area below
threshold). CUB labels are species-constant, so a 'tail pattern' attribute is
marked present even on photos where the tail is hidden; this corrects that.

Outputs (under --data-root):
  CUB_processed/<out-dir>/{train,val,test}.pkl   # evaluation-label copies
  cub70_relabel_diagnostics.parquet              # one row per (image, attribute):
        original_label, part, coarse_area_frac, coarse_visible, new_label, flipped

IMPORTANT CONSTRAINT: CUB70 masks exist only for TEST images. This script may be
used to create visibility-aware evaluation labels, but it cannot create a
visibility-aware training set. Train/val records are copied unchanged. Do not
describe an original-vs-relabeled CUB70 comparison as a retraining intervention.
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cub70_parts import attribute_to_part, COARSE_TO_CUB70

CUB_USED_ATTRIBUTE_IDS = [
    1,4,6,7,10,14,15,20,21,23,25,29,30,35,36,38,40,44,45,50,51,53,54,56,57,59,
    63,64,69,70,72,75,80,84,90,91,93,99,101,106,110,111,116,117,119,125,126,
    131,132,134,145,149,151,152,153,157,158,163,164,168,172,178,179,181,183,
    187,188,193,194,196,198,202,203,208,209,211,212,213,218,220,221,225,235,
    236,238,239,240,242,243,244,249,253,254,259,260,262,268,274,277,283,289,
    292,293,294,298,299,304,305,308,309,310,311,
]


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
        attrs_txt = Path(attr_names_file)
    else:
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
        # Also accept an already-filtered file containing exactly 112 bare names.
        names = [line.strip() for line in attrs_txt.read_text().splitlines()
                 if line.strip()]
        if len(names) == len(CUB_USED_ATTRIBUTE_IDS):
            return names
        raise ValueError(
            f"Could not parse numbered attributes from {attrs_txt}; "
            f"an unnumbered file must contain exactly 112 names, got {len(names)}"
        )

    # class_attr_data_10 contains the official 112 selected attributes, in this
    # exact order (ConceptBottleneck/CUB/generate_new_data.py).
    missing = [i for i in CUB_USED_ATTRIBUTE_IDS if i not in attrs_by_id]
    if missing:
        raise ValueError(f"attributes.txt is missing selected IDs: {missing[:10]}")
    return [attrs_by_id[i] for i in CUB_USED_ATTRIBUTE_IDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--data-dir", default="CUB_processed/class_attr_data_10")
    ap.add_argument("--out-dir", default="CUB_processed/class_attr_data_10_cub70_eval_relabeled")
    ap.add_argument("--attr-names", default=None,
                    help="text file, one attribute name per line (optional; defaults to "
                         "canonical CUB location from official release)")
    ap.add_argument("--cub-root", default=None,
                    help="official CUB_200_2011 root; default: $CURATED_DATA/CUB_200_2011")
    ap.add_argument("--vis-threshold", type=float, default=0.001)
    args = ap.parse_args()
    root = Path(args.data_root)
    src = root / args.data_dir
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Load attribute names with fallback to canonical CUB location
    cub_root = Path(args.cub_root) if args.cub_root else (root / "CUB_200_2011" if root else None)
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
        if recs and len(recs[0]["attribute_label"]) != len(attr_names):
            raise ValueError(f"{f}: {len(recs[0]['attribute_label'])} labels but "
                             f"{len(attr_names)} selected attribute names")
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
    changed_splits = sorted(diag.loc[diag.flipped == 1, "split"].unique()) if considered else []
    if any(s != "test" for s in changed_splits):
        raise RuntimeError(f"CUB70 unexpectedly changed non-test splits: {changed_splits}")
    print("NOTE: masks cover test images only; this output changes evaluation labels, not training labels.")


if __name__ == "__main__":
    main()
