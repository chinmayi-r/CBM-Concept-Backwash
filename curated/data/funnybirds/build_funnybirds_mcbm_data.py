"""
curated/data/funnybirds/build_funnybirds_mcbm_data.py

Build metadata CSVs expected by the MCBM trainer (antonioalmudevar/minimal_cbm)
from the raw FunnyBirds dataset.

The MCBM trainer expects one CSV per split with columns:
    image_path, label, concept_0, concept_1, ..., concept_{N-1}

Outputs (to $CURATED_DATA/funnybirds_mcbm/):
    train.csv, test.csv

Usage:
    python build_funnybirds_mcbm_data.py \\
        --funnybirds_root /path/to/FunnyBirds \\
        --out_dir         $CURATED_DATA/funnybirds_mcbm \\
        [--concept_labels /path/to/concept_labels_image_level.npy]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.funnybirds_dataset import FunnyBirdsDataset, NUM_CONCEPTS, concept_names


def build_split(funnybirds_root: Path, split: str, concept_labels_path: Path | None) -> pd.DataFrame:
    ds = FunnyBirdsDataset(
        funnybirds_root=funnybirds_root,
        split=split,
        transform=None,
        include_concepts=True,
        concept_labels_path=concept_labels_path if split == "train" else None,
    )
    cnames = concept_names()
    rows = []
    for idx in range(len(ds)):
        sample = ds[idx]
        class_idx = int(sample["label"].item())
        img_path = funnybirds_root / split / str(class_idx) / f"{idx:06d}.png"
        concepts = sample["concepts"].tolist()
        row = {"image_path": str(img_path.resolve()), "label": class_idx}
        for name, val in zip(cnames, concepts):
            row[name] = int(val)
        rows.append(row)
        if idx % 5000 == 0:
            print(f"  {split}: {idx}/{len(ds)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds_root", required=True, type=Path)
    ap.add_argument("--out_dir",         required=True, type=Path)
    ap.add_argument("--concept_labels",  default=None,  type=Path,
                    help="optional (N_train, 26) .npy from make_image_level_concept_labels.py")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        print(f"Building {split}…")
        df = build_split(args.funnybirds_root, split, args.concept_labels)
        out = args.out_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        print(f"  → {out}  ({len(df)} rows, {NUM_CONCEPTS} concept cols)")


if __name__ == "__main__":
    main()
