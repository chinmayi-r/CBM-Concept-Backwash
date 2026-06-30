"""
curated/data/cub/build_cub_mcbm_data.py

Build metadata CSVs expected by the MCBM trainer (antonioalmudevar/minimal_cbm)
from the standard CUB-200-2011 dataset, using the same 112-attribute subset as
the CBM paper.

Outputs (to $CURATED_DATA/cub_mcbm/):
    train.csv, test.csv

Usage:
    python build_cub_mcbm_data.py \\
        --cub_root /path/to/CUB_200_2011 \\
        --out_dir  $CURATED_DATA/cub_mcbm
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.cub_metadata import load_cub_metadata
from curated.data.cub.build_cub_cbm_data import SELECTED_ATTR_IDS, N_ATTR


def build_split(cub_root: Path, split: str) -> pd.DataFrame:
    meta = load_cub_metadata(cub_root)

    images = meta.images.set_index("image_id")
    labels = meta.image_class_labels.set_index("image_id")
    splits = meta.train_test_split.set_index("image_id")

    if meta.image_attributes_binary is None:
        raise RuntimeError("CUB attribute annotations not found.")
    iab = meta.image_attributes_binary.set_index("image_id")

    attr_cols = [f"attr_{aid}_present" for aid in SELECTED_ATTR_IDS]

    # Build attribute names from attribute definitions if available
    if meta.attributes is not None:
        attrs_df = meta.attributes.set_index("attribute_id")
        col_names = [attrs_df.loc[aid, "attribute_name"].replace("::", "_")
                     if aid in attrs_df.index else f"attr_{aid}"
                     for aid in SELECTED_ATTR_IDS]
    else:
        col_names = [f"attr_{aid}" for aid in SELECTED_ATTR_IDS]

    is_train = splits["is_training_image"] == 1
    ids = splits.index[is_train if split == "train" else ~is_train].tolist()

    rows = []
    for img_id in ids:
        cls_id = int(labels.loc[img_id, "class_id"]) - 1
        rel_path = images.loc[img_id, "file_path"]
        img_path = cub_root / "images" / rel_path
        attr_vec = [int(iab.loc[img_id, c]) if img_id in iab.index else 0
                    for c in attr_cols]
        row = {"image_path": str(img_path.resolve()), "label": cls_id}
        for name, val in zip(col_names, attr_vec):
            row[name] = val
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub_root", required=True, type=Path)
    ap.add_argument("--out_dir",  required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        print(f"Building {split}…")
        df = build_split(args.cub_root, split)
        out = args.out_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        print(f"  → {out}  ({len(df)} rows, {N_ATTR} concept cols)")


if __name__ == "__main__":
    main()
