#!/usr/bin/env python3
"""Filter official CUB pickles to classes 0..69 for CUB70 model training.

The segmentation masks are test-only. This script therefore filters the
original labels; it does not pretend to create visibility-aware training labels.
"""
import argparse
import os
import pickle
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--source", default="CUB_processed/class_attr_data_10")
    ap.add_argument("--out", default="CUB_processed/class_attr_data_10_cub70_original")
    args = ap.parse_args()
    root = Path(args.data_root)
    src, out = root / args.source, root / args.out
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = src / f"{split}.pkl"
        if not path.exists():
            continue
        records = pickle.loads(path.read_bytes())
        kept = [r for r in records if 0 <= int(r["class_label"]) < 70]
        (out / f"{split}.pkl").write_bytes(pickle.dumps(kept))
        classes = sorted({int(r["class_label"]) for r in kept})
        if classes and classes != list(range(70)):
            raise RuntimeError(f"{split}: expected classes 0..69, got {classes}")
        print(f"{split}: {len(kept)}/{len(records)} images, {len(classes)} classes")
    if not (out / "train.pkl").exists() or not (out / "test.pkl").exists():
        raise RuntimeError("CUB70 filtered train/test pickles were not produced")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
