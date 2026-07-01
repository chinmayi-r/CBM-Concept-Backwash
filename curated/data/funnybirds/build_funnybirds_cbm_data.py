#!/usr/bin/env python3
"""Convert OFFICIAL FunnyBirds into the CBM pickled-list format.

Built directly on the official release files (no hand-written FunnyBirds code):
  <fb>/dataset_{train,test}.json     per-image params incl. class_idx + part models
  <fb>/parts.json                    part variant definitions (concept schema)
  <fb>/{mode}/{class_idx}/{idx:06d}.png            input images
  <fb>/{mode}_part_map/{class_idx}/{idx:06d}.png   part-map segmentation (visibility)

Paths and the part-map color map match funnybirds-framework/datasets/funny_birds.py.

KNOWN COMPATIBILITY GAP (read before running):
  CUB/dataset.py's image loader is hardcoded to CUB conventions: when the
  "CUB_200_2011" token isn't found in img_path, it falls into a broken except
  branch that reconstructs a path assuming a <root>/<class>/<split>/<file>
  layout, which does not match FunnyBirds' <root>/<mode>/<class>/<idx>.png
  layout, and Image.open() raises when it can't find the reconstructed path.
  Per curated/README.md, nothing in external/ may be edited, so this builder
  works around it the same way build_cub_cbm_data.py's real CUB pipeline is
  laid out: it creates a symlink named CUB_200_2011 pointing at
  `funnybirds_root` inside `out_dir`, and writes img_path through that
  symlink. Pass `-image_dir <out_dir>/CUB_200_2011` to experiments.py
  (curated/train/cbm_funnybirds.sh does this) so the token search succeeds.
  This is a path-construction shim only -- no FunnyBirds-specific behavior is
  added to the official trainer.

  CUB/train.py also hardcodes `val_data_path = train_data_path.replace(
  'train.pkl', 'val.pkl')` and will crash with FileNotFoundError if val.pkl is
  missing -- this builder carves a seeded 20% val split off the official train
  images (CUB/data_processing.py's own val_ratio=0.2), same as
  build_cub_cbm_data.py, since FunnyBirds ships only a 2-way
  dataset_train.json/dataset_test.json split.

Outputs:
  $CURATED_DATA/funnybirds_processed/{train,val,test}.pkl   (CUBDataset schema)
  $CURATED_DATA/funnybirds_processed/CUB_200_2011           (symlink -> funnybirds_root)
  $CURATED_DATA/funnybirds_visibility.parquet                (per image x coarse part)

--labels species      : species-level one-hot from params (default)
--labels image_level  : zero a part's concept group when that part is occluded in
                        the image (coarse part pixels < threshold x class-median),
                        i.e. visibility-aware labels from the OFFICIAL part maps
                        (prof note #1 for FunnyBirds).

The `image` key (joins eval tables <-> visibility) is the relative path
"{mode}/{class_idx}/{idx:06d}.png", unique across the dataset.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
    def _read_rgb(p): return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
except ImportError:  # pragma: no cover
    from PIL import Image
    def _read_rgb(p): return np.array(Image.open(p).convert("RGB"))

from funnybirds_concepts import (
    load_parts, build_part_lookup, params_to_concept_vector, concept_names,
    group_slices, PARTMAP_COLOR_TO_INSTANCE, INSTANCE_TO_COARSE, COARSE_PARTS,
)


def coarse_pixel_counts(part_map: np.ndarray) -> dict:
    """Count pixels per coarse part from an official part-map PNG."""
    counts = defaultdict(int)
    flat = part_map.reshape(-1, 3)
    for color, inst in PARTMAP_COLOR_TO_INSTANCE.items():
        n = int(np.all(flat == np.array(color, dtype=flat.dtype), axis=1).sum())
        counts[INSTANCE_TO_COARSE[inst]] += n
    return {p: counts.get(p, 0) for p in COARSE_PARTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--funnybirds-root", required=True,
                    help="official FunnyBirds dataset root (has dataset_*.json, parts.json, part maps)")
    ap.add_argument("--labels", choices=["species", "image_level"], default="species")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="image_level: visible if coarse pixels >= threshold * class-median")
    ap.add_argument("--no-visibility", action="store_true",
                    help="skip reading part maps (faster; disables visibility table & image_level)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="fraction of official train images held out as val (CUB/data_processing.py's own default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_root)
    fb = Path(args.funnybirds_root)
    out = root / "funnybirds_processed"
    out.mkdir(parents=True, exist_ok=True)

    symlink_root = out / "CUB_200_2011"
    if not symlink_root.exists():
        symlink_root.symlink_to(fb.resolve(), target_is_directory=True)
        print(f"  created {symlink_root} -> {fb.resolve()}")

    parts = load_parts(fb)
    lut = build_part_lookup(parts)
    names = concept_names(parts)
    spans = group_slices(parts)
    print(f"  parts.json -> {len(names)} concepts: {dict((p, len(v)) for p, v in parts.items())}")

    need_partmaps = (not args.no_visibility) or args.labels == "image_level"
    vis_rows = []
    rec_id = 0

    for mode in ("train", "test"):
        dj = fb / f"dataset_{mode}.json"
        if not dj.exists():
            print(f"  {dj} missing, skipping {mode}")
            continue
        params = json.loads(dj.read_text())

        # pass 1 (image_level only): coarse pixel counts + per-class medians
        pix = {}
        if need_partmaps:
            for idx, entry in enumerate(params):
                c = entry["class_idx"]
                pm = fb / f"{mode}_part_map" / str(c) / f"{idx:06d}.png"
                pix[idx] = coarse_pixel_counts(_read_rgb(pm)) if pm.exists() else {p: 0 for p in COARSE_PARTS}
        medians = {}
        if args.labels == "image_level":
            byclass = defaultdict(lambda: defaultdict(list))
            for idx, entry in enumerate(params):
                for p in COARSE_PARTS:
                    byclass[entry["class_idx"]][p].append(pix[idx][p])
            medians = {c: {p: float(np.median(v)) for p, v in d.items()} for c, d in byclass.items()}

        recs = []
        for idx, entry in enumerate(params):
            c = entry["class_idx"]
            rel = f"{mode}/{c}/{idx:06d}.png"
            attr = params_to_concept_vector(parts, lut, entry)

            if args.labels == "image_level":
                for p, (a, b) in spans.items():
                    med = medians[c].get(p, 0.0)
                    visible = pix[idx][p] >= args.threshold * med if med > 0 else pix[idx][p] > 0
                    if not visible:
                        for j in range(a, b):
                            attr[j] = 0

            # img_path goes through the CUB_200_2011 symlink so the official
            # CUB/dataset.py path-token search succeeds unmodified (see
            # module docstring); "image" is the raw relative path, used to
            # join against funnybirds_visibility.parquet / eval tables.
            recs.append({
                "id": rec_id, "img_path": str(symlink_root / rel), "image": rel,
                "class_label": int(c),
                "attribute_label": attr, "attribute_certainty": [4] * len(attr),
            })
            if need_partmaps:
                for p in COARSE_PARTS:
                    px = pix[idx][p]
                    vis_rows.append({"image_name": rel, "mode": mode, "part": p,
                                     "pixel_count": px, "visible": px > 0})
            rec_id += 1

        if mode == "train":
            # Official FunnyBirds ships only train/test; carve val off train
            # the same seeded way build_cub_cbm_data.py does, since
            # CUB/train.py hardcodes a val.pkl load.
            rng = random.Random(args.seed)
            order = list(range(len(recs)))
            rng.shuffle(order)
            n_val = int(args.val_ratio * len(order))
            val_idx, train_idx = set(order[:n_val]), set(order[n_val:])
            train_recs = [recs[i] for i in sorted(train_idx)]
            val_recs = [recs[i] for i in sorted(val_idx)]
            with open(out / "train.pkl", "wb") as f:
                pickle.dump(train_recs, f)
            with open(out / "val.pkl", "wb") as f:
                pickle.dump(val_recs, f)
            print(f"  wrote train.pkl: {len(train_recs)} images, val.pkl: {len(val_recs)} images")
        else:
            with open(out / f"{mode}.pkl", "wb") as f:
                pickle.dump(recs, f)
            print(f"  wrote {mode}.pkl: {len(recs)} images")

    if vis_rows:
        pd.DataFrame(vis_rows).to_parquet(root / "funnybirds_visibility.parquet", index=False)
        print(f"  wrote funnybirds_visibility.parquet: {len(vis_rows)} rows")


if __name__ == "__main__":
    main()
