#!/usr/bin/env python
"""
scripts/prepare_funnybirds_metadata.py

Parse the FunnyBirds annotation.json and emit CSV files that mirror the CUB
metadata structure used elsewhere in this pipeline.

Usage:
    python scripts/prepare_funnybirds_metadata.py \\
        --funnybirds_root data/FunnyBirds \\
        --out_dir data/FunnyBirds/metadata

Output files (schema mirrors CUB prepare_cub_metadata.py output):
    images.csv                   image_id, file_path, is_train, class_id
    classes.csv                  class_id, class_name
    concepts.csv                 concept_id, concept_name, part, variant
    image_concepts_binary.csv    image_id, beak_0, ..., tail_8   (26 cols)
    class_concept_matrix.csv     class_id, beak_0, ..., tail_8   (ground truth)

The class_concept_matrix.csv is the key FunnyBirds advantage: it is the exact,
noise-free species-concept mapping that CUB lacks.

FunnyBirds annotation.json format (one dict per image):
    {
        "id":        0,
        "file_name": "0_0000.png",
        "class":     0,          # 0-based class index
        "beak":      2,          # variant index (0-based)
        "eye":       0,
        "wing":      3,
        "foot":      1,
        "tail":      5,
        "bg":        4,          # background (not used as concept)
        "split":     "train"
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datasets.funnybirds_dataset import PARTS, PART_VARIANTS, concept_names


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FunnyBirds metadata CSVs mirroring CUB schema"
    )
    parser.add_argument("--funnybirds_root", type=str, required=True,
                        help="Path to FunnyBirds root (contains data/ and annotation.json)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for CSVs (default: <funnybirds_root>/metadata)")
    args = parser.parse_args()

    root = Path(args.funnybirds_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = root / "annotation.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation.json not found at {ann_path}")

    with open(ann_path) as f:
        annotations = json.load(f)

    print(f"[prepare_funnybirds] Loaded {len(annotations)} annotations from {ann_path}")

    cnames = concept_names()  # 26 concept column names

    # ── images.csv ────────────────────────────────────────────────────────────
    image_rows = []
    for ann in annotations:
        image_rows.append({
            "image_id": int(ann["id"]),
            "file_path": str(ann["file_name"]),
            "class_id":  int(ann["class"]),
            "is_train":  1 if ann.get("split", "train") == "train" else 0,
        })
    images_df = pd.DataFrame(image_rows).sort_values("image_id").reset_index(drop=True)
    images_df.to_csv(out_dir / "images.csv", index=False)
    print(f"[prepare_funnybirds] Wrote images.csv  ({len(images_df)} rows)")

    # ── classes.csv ───────────────────────────────────────────────────────────
    class_ids = sorted(images_df["class_id"].unique())
    classes_df = pd.DataFrame({
        "class_id":   class_ids,
        "class_name": [f"funnybird_{c:02d}" for c in class_ids],
    })
    classes_df.to_csv(out_dir / "classes.csv", index=False)
    print(f"[prepare_funnybirds] Wrote classes.csv ({len(classes_df)} classes)")

    # ── concepts.csv ──────────────────────────────────────────────────────────
    concept_rows = []
    cid = 0
    for part, n_var in PART_VARIANTS.items():
        for v in range(n_var):
            concept_rows.append({
                "concept_id":   cid,
                "concept_name": f"{part}_{v}",
                "part":         part,
                "variant":      v,
            })
            cid += 1
    concepts_df = pd.DataFrame(concept_rows)
    concepts_df.to_csv(out_dir / "concepts.csv", index=False)
    print(f"[prepare_funnybirds] Wrote concepts.csv ({len(concepts_df)} concepts)")

    # ── image_concepts_binary.csv ─────────────────────────────────────────────
    concept_rows = []
    for ann in annotations:
        row = {"image_id": int(ann["id"])}
        for part, n_var in PART_VARIANTS.items():
            v = int(ann[part])
            for vi in range(n_var):
                row[f"{part}_{vi}"] = 1 if vi == v else 0
        concept_rows.append(row)

    img_concepts_df = (
        pd.DataFrame(concept_rows)
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    img_concepts_df.to_csv(out_dir / "image_concepts_binary.csv", index=False)
    print(f"[prepare_funnybirds] Wrote image_concepts_binary.csv ({len(img_concepts_df)} rows)")

    # ── class_concept_matrix.csv  (ground truth, unique to FunnyBirds) ────────
    # Each class maps to exactly one part combination; derive from any image of that class.
    class_rows = {}
    for ann in annotations:
        c = int(ann["class"])
        if c not in class_rows:
            row = {"class_id": c}
            for part, n_var in PART_VARIANTS.items():
                v = int(ann[part])
                for vi in range(n_var):
                    row[f"{part}_{vi}"] = 1 if vi == v else 0
            class_rows[c] = row

    class_concept_df = (
        pd.DataFrame(list(class_rows.values()))
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    class_concept_df.to_csv(out_dir / "class_concept_matrix.csv", index=False)
    print(f"[prepare_funnybirds] Wrote class_concept_matrix.csv ({len(class_concept_df)} classes)")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_train = (images_df["is_train"] == 1).sum()
    n_test  = (images_df["is_train"] == 0).sum()
    print(f"\n[prepare_funnybirds] Summary:")
    print(f"  Classes:    {len(class_ids)}")
    print(f"  Concepts:   {len(cnames)} binary (one-hot over part variants)")
    print(f"  Train imgs: {n_train}")
    print(f"  Test imgs:  {n_test}")
    print(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
