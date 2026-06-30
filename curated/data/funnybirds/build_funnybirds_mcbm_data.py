#!/usr/bin/env python3
"""Convert FunnyBirds into the minimal_cbm (MCBM) manifest, from the CBM pickles.

Reads $CURATED_DATA/funnybirds_processed/{train,test}.pkl (produced by
build_funnybirds_cbm_data.py from the official files) and writes:
  $CURATED_DATA/funnybirds_mcbm/{train,test}.csv   # img_path, image, class_idx, c0..c{N-1}
  $CURATED_DATA/funnybirds_mcbm/concepts.json      # names + group spans from parts.json

Pair with train/configs/funnybirds-mcbm.yaml (its data section points here).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from funnybirds_concepts import load_parts, concept_names, group_slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--funnybirds-root", required=True,
                    help="official FunnyBirds root (for parts.json -> concept names)")
    args = ap.parse_args()
    root = Path(args.data_root)
    src = root / "funnybirds_processed"
    out = root / "funnybirds_mcbm"
    out.mkdir(parents=True, exist_ok=True)

    parts = load_parts(args.funnybirds_root)
    names = concept_names(parts)
    n = len(names)
    cols = ["img_path", "image", "class_idx"] + [f"c{i}" for i in range(n)]

    for split in ("train", "test"):
        f = src / f"{split}.pkl"
        if not f.exists():
            continue
        recs = pickle.loads(f.read_bytes())
        rows = [[r["img_path"], r["image"], r["class_label"], *r["attribute_label"]] for r in recs]
        assert all(len(r["attribute_label"]) == n for r in recs), "concept width mismatch vs parts.json"
        pd.DataFrame(rows, columns=cols).to_csv(out / f"{split}.csv", index=False)
        print(f"  wrote {split}.csv: {len(rows)} rows, {n} concepts")

    (out / "concepts.json").write_text(json.dumps({
        "names": names,
        "group_slices": {k: list(v) for k, v in group_slices(parts).items()},
        "n_concepts": n,
    }, indent=2))
    print(f"  wrote concepts.json ({n} concepts)")


if __name__ == "__main__":
    main()
