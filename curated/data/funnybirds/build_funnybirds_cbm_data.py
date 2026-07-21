#!/usr/bin/env python3
"""Convert OFFICIAL FunnyBirds into the CBM pickled-list format.

Built directly on the official release files (no hand-written FunnyBirds code):
  <fb>/dataset_{train,test}.json     per-image params incl. class_idx + part models
  <fb>/parts.json                    part variant definitions (concept schema)
  <fb>/{mode}/{class_idx}/{idx:06d}.png            input images
  <fb>/{mode}_part_map/{class_idx}/{idx:06d}.png   part-map segmentation (visibility)

Paths and the part-map color map match funnybirds-framework/datasets/funny_birds.py.

Outputs:
  $CURATED_DATA/funnybirds_processed/{train,test}.pkl   (CUBDataset schema)
  $CURATED_DATA/funnybirds_visibility.parquet           (per image x coarse part)

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
    PARTMAP_COLOR_TO_INSTANCE, INSTANCE_TO_COARSE, COARSE_PARTS,
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
    ap.add_argument("--funnybirds-root",
                    default=str(Path(os.environ.get("CURATED_DATA", "")) / "FunnyBirds"),
                    help="official FunnyBirds root; default: $CURATED_DATA/FunnyBirds")
    ap.add_argument("--labels", choices=["species", "image_level"], default="species")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="image_level: visible if coarse pixels >= threshold * class-median")
    ap.add_argument("--no-visibility", action="store_true",
                    help="skip reading part maps (faster; disables visibility table & image_level)")
    ap.add_argument("--out-name", "--output-subdir", dest="out_name",
                    default="funnybirds_processed",
                    help="output subdir under data-root; use funnybirds_processed_rl for the "
                         "relabeled (image_level) build so it does NOT overwrite the standard pkls")
    args = ap.parse_args()

    root = Path(args.data_root)
    fb = Path(args.funnybirds_root)
    out = root / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    parts = load_parts(fb)
    lut = build_part_lookup(parts)
    names = concept_names(parts)
    print(f"  parts.json -> {len(names)} concepts: {dict((p, len(v)) for p, v in parts.items())}")

    need_partmaps = (not args.no_visibility) or args.labels == "image_level"
    vis_rows = []
    splits = {}
    rec_id = 0

    for mode in ("train", "test"):
        dj = fb / f"dataset_{mode}.json"
        if not dj.exists():
            print(f"  {dj} missing, skipping {mode}")
            continue
        params = json.loads(dj.read_text())

        # pass 1 (image_level only): coarse pixel counts + per-class medians
        pix = {}
        missing_partmaps = []
        if need_partmaps:
            for idx, entry in enumerate(params):
                c = entry["class_idx"]
                pm = fb / f"{mode}_part_map" / str(c) / f"{idx:06d}.png"
                if not pm.exists():
                    missing_partmaps.append(pm)
                else:
                    pix[idx] = coarse_pixel_counts(_read_rgb(pm))
                if (idx + 1) % 5000 == 0:
                    print(f"  {mode}: scanned {idx + 1}/{len(params)} part maps", flush=True)
            if missing_partmaps:
                examples = "\n".join(f"    {p}" for p in missing_partmaps[:5])
                raise FileNotFoundError(
                    f"{len(missing_partmaps)} {mode} part maps are missing; refusing "
                    f"to relabel them as fully occluded. First examples:\n{examples}"
                )
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
                spans = _group_spans(parts)
                for p, (a, b) in spans.items():
                    med = medians[c].get(p, 0.0)
                    visible = pix[idx][p] >= args.threshold * med if med > 0 else pix[idx][p] > 0
                    if not visible:
                        for j in range(a, b):
                            attr[j] = 0

            recs.append({
                "id": rec_id, "img_path": str(fb / rel), "image": rel,
                "class_label": int(c),
                "attribute_label": attr, "attribute_certainty": [4] * len(attr),
            })
            if need_partmaps:
                for p in COARSE_PARTS:
                    px = pix[idx][p]
                    vis_rows.append({"image_name": rel, "mode": mode, "part": p,
                                     "pixel_count": px, "visible": px > 0})
            rec_id += 1
        splits[mode] = recs
        with open(out / f"{mode}.pkl", "wb") as f:
            pickle.dump(recs, f)
        print(f"  wrote {mode}.pkl: {len(recs)} images")

    if vis_rows:
        pd.DataFrame(vis_rows).to_parquet(root / "funnybirds_visibility.parquet", index=False)
        print(f"  wrote funnybirds_visibility.parquet: {len(vis_rows)} rows")


def _group_spans(parts):
    from funnybirds_concepts import group_slices
    # collapse per-variant groups to coarse parts (here parts == coarse already)
    return group_slices(parts)


if __name__ == "__main__":
    main()
