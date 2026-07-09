#!/usr/bin/env python3
"""Standard val-based model selection for the OFFICIAL minimal_cbm trainer.

minimal_cbm reads only train.pkl/test.pkl and evaluates on 'test' EVERY epoch, so
selecting a checkpoint by that number would leak the test set. The standard CBM
protocol (Koh et al. 2020) instead selects on a VALIDATION set and touches test
once. To do that WITHOUT editing the submodule, we build a sibling dir:

    <pkls>_trainval/train.pkl = training data MINUS the val fold
    <pkls>_trainval/test.pkl  = the val fold      <-- trainer's per-epoch eval == VAL

Training points pkls_dir at <pkls>_trainval (train/_paths.sh does this
automatically when the dir exists). The REAL test set stays in the source dir and
is used ONCE, at final eval of the val-selected checkpoint.

If the source already has an official val.pkl (e.g. CUB_processed), we use it
verbatim instead of re-splitting.

Usage:
    python data/make_val_split.py --pkls-dir $CURATED_DATA/funnybirds_processed
    python data/make_val_split.py --pkls-dir $CURATED_DATA/CUB_processed/class_attr_data_10
"""
from __future__ import annotations
import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path


def _load(p):
    with open(p, "rb") as f: return pickle.load(f)

def _dump(o, p):
    with open(p, "wb") as f: pickle.dump(o, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls-dir", required=True, help="dir with train.pkl (+ test.pkl, maybe val.pkl)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of train per class held out as val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None, help="default: <pkls-dir>_trainval")
    args = ap.parse_args()

    src = Path(args.pkls_dir)
    out = Path(args.out_dir) if args.out_dir else Path(str(src) + "_trainval")
    out.mkdir(parents=True, exist_ok=True)
    train = _load(src / "train.pkl")

    if (src / "val.pkl").exists():
        val = _load(src / "val.pkl")
        new_train = train
        how = f"used existing official val.pkl ({len(val)} images)"
    else:
        by_class = defaultdict(list)
        for i, r in enumerate(train):
            by_class[r["class_label"]].append(i)
        rng = random.Random(args.seed)
        val_idx = set()
        for _, idxs in by_class.items():
            k = max(1, int(round(len(idxs) * args.val_frac)))
            val_idx.update(rng.sample(idxs, k))
        val = [train[i] for i in sorted(val_idx)]
        new_train = [r for i, r in enumerate(train) if i not in val_idx]
        how = (f"carved {len(val)} val from {len(train)} train "
               f"(val_frac={args.val_frac}, per-class stratified, seed={args.seed})")

    _dump(new_train, out / "train.pkl")
    _dump(val,       out / "test.pkl")   # trainer's per-epoch eval == VAL
    print(f"[make_val_split] {how}")
    print(f"  wrote {out}/train.pkl ({len(new_train)})  and  {out}/test.pkl (=VAL, {len(val)})")
    print(f"  REAL test stays at {src}/test.pkl -- touched ONCE, at final eval of the val-selected ckpt.")


if __name__ == "__main__":
    main()
