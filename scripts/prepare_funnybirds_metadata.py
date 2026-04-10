#!/usr/bin/env python
"""
scripts/prepare_funnybirds_metadata.py

Parse the FunnyBirds dataset_train.json / dataset_test.json and emit CSV files
that mirror the CUB metadata structure used elsewhere in this pipeline.

Usage:
    python -m scripts.prepare_funnybirds_metadata \\
        --funnybirds_root data/FunnyBirds \\
        --out_dir data/FunnyBirds/metadata

Dataset download (if you haven't already):
    wget https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
    unzip FunnyBirds.zip
    mv FunnyBirds data/FunnyBirds

Output files (schema mirrors prepare_cub_metadata.py output):
    images.csv                   image_id, file_path, is_train, class_id
    classes.csv                  class_id, class_name
    concepts.csv                 concept_id, concept_name, part, variant
    image_concepts_binary.csv    image_id, beak_0, ..., tail_8  (27 cols total)
    class_concept_matrix.csv     class_id, beak_0, ..., tail_8  (ground truth)

Image ID convention (globally unique across splits):
    train images: image_id = idx  (0 .. N_train-1)
    test  images: image_id = _FUNNYBIRDS_N_TRAIN + idx  (N_train .. N_train+N_test-1)
This matches what FunnyBirdsDataset.__getitem__ returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from datasets.funnybirds_dataset import (
    PART_VARIANTS,
    concept_names,
    _build_part_lookup,
    _params_to_variant_idx,
    _FUNNYBIRDS_N_TRAIN,
)


def _ann_to_concept_row(image_id: int, ann: dict, lookup: dict) -> dict:
    """Build one row of image_concepts_binary.csv for a single annotation entry."""
    row: dict = {"image_id": image_id}
    for part, n_var in PART_VARIANTS.items():
        v = _params_to_variant_idx(lookup, part, ann)
        for vi in range(n_var):
            row[f"{part}_{vi}"] = 1 if vi == v else 0
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FunnyBirds metadata CSVs mirroring CUB schema"
    )
    parser.add_argument(
        "--funnybirds_root", type=str, required=True,
        help="Path to FunnyBirds root (contains dataset_train.json, parts.json, etc.)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for CSVs (default: <funnybirds_root>/metadata)"
    )
    args = parser.parse_args()

    root = Path(args.funnybirds_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load JSONs ────────────────────────────────────────────────────────────
    for fname in ("dataset_train.json", "dataset_test.json", "parts.json"):
        if not (root / fname).exists():
            raise FileNotFoundError(
                f"{root / fname} not found.\n"
                "Download the FunnyBirds dataset (not the code repo):\n"
                "  wget https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip\n"
                "  unzip FunnyBirds.zip && mv FunnyBirds <funnybirds_root>"
            )

    with open(root / "dataset_train.json") as f:
        train_anns = json.load(f)
    with open(root / "dataset_test.json") as f:
        test_anns = json.load(f)
    with open(root / "parts.json") as f:
        parts_json = json.load(f)

    lookup = _build_part_lookup(parts_json)
    cnames = concept_names()  # 26 concept column names

    print(f"[prepare_funnybirds] Loaded {len(train_anns)} train / {len(test_anns)} test annotations")

    # ── images.csv ────────────────────────────────────────────────────────────
    image_rows = []
    for idx, ann in enumerate(train_anns):
        c = int(ann["class_idx"])
        image_rows.append({
            "image_id":  idx,
            "file_path": f"train/{c}/{idx:06d}.png",
            "is_train":  1,
            "class_id":  c,
        })
    for idx, ann in enumerate(test_anns):
        c = int(ann["class_idx"])
        image_rows.append({
            "image_id":  _FUNNYBIRDS_N_TRAIN + idx,
            "file_path": f"test/{c}/{idx:06d}.png",
            "is_train":  0,
            "class_id":  c,
        })
    images_df = pd.DataFrame(image_rows)
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
    concept_rows2 = []
    for idx, ann in enumerate(train_anns):
        concept_rows2.append(_ann_to_concept_row(idx, ann, lookup))
    for idx, ann in enumerate(test_anns):
        concept_rows2.append(_ann_to_concept_row(_FUNNYBIRDS_N_TRAIN + idx, ann, lookup))

    img_concepts_df = pd.DataFrame(concept_rows2)
    img_concepts_df.to_csv(out_dir / "image_concepts_binary.csv", index=False)
    print(f"[prepare_funnybirds] Wrote image_concepts_binary.csv ({len(img_concepts_df)} rows)")

    # ── class_concept_matrix.csv  (ground truth, unique to FunnyBirds) ────────
    # Each class maps to exactly one part combination; derive from first seen image.
    class_rows: dict = {}
    for idx, ann in enumerate(train_anns):
        c = int(ann["class_idx"])
        if c not in class_rows:
            row = _ann_to_concept_row(c, ann, lookup)
            row["image_id"] = c  # reuse slot; will be renamed class_id below
            class_rows[c] = row

    class_concept_df = (
        pd.DataFrame(list(class_rows.values()))
        .rename(columns={"image_id": "class_id"})
        .sort_values("class_id")
        .reset_index(drop=True)
    )
    class_concept_df.to_csv(out_dir / "class_concept_matrix.csv", index=False)
    print(f"[prepare_funnybirds] Wrote class_concept_matrix.csv ({len(class_concept_df)} classes)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[prepare_funnybirds] Summary:")
    print(f"  Classes:    {len(class_ids)}")
    print(f"  Concepts:   {len(cnames)} binary (one-hot over part variants)")
    print(f"  Train imgs: {len(train_anns)}")
    print(f"  Test imgs:  {len(test_anns)}")
    print(f"  Test image_id range: [{_FUNNYBIRDS_N_TRAIN}, {_FUNNYBIRDS_N_TRAIN + len(test_anns) - 1}]")
    print(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
