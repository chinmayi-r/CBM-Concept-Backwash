#!/usr/bin/env python3
"""Build the MCBM CSV manifest for CUB from the official processed pickles.

Reads $CURATED_DATA/CUB_processed/class_attr_data_10/{train,val,test}.pkl and
writes $CURATED_DATA/cub_mcbm/{train,val,test}.csv with columns
img_path, class_idx, c0..c111 -- so CBM and MCBM train on identical labels.

Pass --data-dir to relabel against a visibility-corrected pickle dir (e.g.
class_attr_data_10_relabeled) when training the relabeled MCBM ablation.
"""
from __future__ import annotations
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--data-dir", default="CUB_processed/class_attr_data_10",
                    help="pickle dir relative to data-root")
    ap.add_argument("--out", default="cub_mcbm",
                    help="output dir relative to data-root")
    args = ap.parse_args()
    root = Path(args.data_root)
    src = root / args.data_dir
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    n_attr = None
    for split in ("train", "val", "test"):
        recs = pickle.loads((src / f"{split}.pkl").read_bytes())
        n_attr = len(recs[0]["attribute_label"])
        cols = ["img_path", "class_idx"] + [f"c{i}" for i in range(n_attr)]
        rows = [[r["img_path"], r["class_label"], *r["attribute_label"]] for r in recs]
        pd.DataFrame(rows, columns=cols).to_csv(out / f"{split}.csv", index=False)
        print(f"  wrote {split}.csv: {len(rows)} rows, {n_attr} concepts")


if __name__ == "__main__":
    main()
