"""
curated/data/cub/build_cub_cbm_data.py

Build the pickled list-of-dicts expected by the official CBM trainer
(yewsiang/ConceptBottleneck) from the standard CUB-200-2011 dataset.

The 112 binary attributes used in the CBM paper are a hand-selected subset of
CUB's 312 attributes, listed in external/ConceptBottleneck/CUB/data_utils.py.
We re-use that list directly to stay faithful (N_ATTR = 112).

Outputs (to $CURATED_DATA/cub_processed/):
    train.pkl, test.pkl

Usage:
    python build_cub_cbm_data.py \\
        --cub_root /path/to/CUB_200_2011 \\
        --out_dir  $CURATED_DATA/cub_processed
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.cub_metadata import load_cub_metadata

# 112 attribute IDs used in Koh et al. 2020 (CBM paper), 1-indexed.
# Source: ConceptBottleneck/CUB/data_utils.py, SELECTED_CONCEPTS list.
SELECTED_ATTR_IDS: list[int] = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45,
    50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93,
    99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149,
    151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188,
    193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225,
    235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268,
    274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311,
]

N_ATTR = len(SELECTED_ATTR_IDS)  # 112


def build_split(cub_root: Path, split: str) -> list[dict]:
    meta = load_cub_metadata(cub_root)

    images = meta.images.set_index("image_id")
    labels = meta.image_class_labels.set_index("image_id")
    splits = meta.train_test_split.set_index("image_id")

    if meta.image_attributes_binary is None:
        raise RuntimeError(
            "CUB attribute annotations not found. "
            "Ensure image_attribute_labels.txt is present under cub_root."
        )
    iab = meta.image_attributes_binary.set_index("image_id")

    attr_cols = [f"attr_{aid}_present" for aid in SELECTED_ATTR_IDS]
    missing = [c for c in attr_cols if c not in iab.columns]
    if missing:
        raise RuntimeError(f"Missing attribute columns: {missing[:5]}…")

    is_train = splits["is_training_image"] == 1
    ids = splits.index[is_train if split == "train" else ~is_train].tolist()

    records = []
    for img_id in ids:
        cls_id = int(labels.loc[img_id, "class_id"]) - 1  # 0-based
        rel_path = images.loc[img_id, "file_path"]
        img_path = cub_root / "images" / rel_path
        attr_vec = [int(iab.loc[img_id, c]) if img_id in iab.index else 0
                    for c in attr_cols]
        records.append({
            "id":                   str(img_id),
            "img_path":             str(img_path.resolve()),
            "class_label":          cls_id,
            "attribute_label":      attr_vec,
            "img_cover":            1.0,
            "attribute_certainty":  [1.0] * N_ATTR,
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub_root", required=True, type=Path)
    ap.add_argument("--out_dir",  required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        print(f"Building {split}…")
        records = build_split(args.cub_root, split)
        out = args.out_dir / f"{split}.pkl"
        with open(out, "wb") as f:
            pickle.dump(records, f)
        print(f"  → {out}  ({len(records)} records, {N_ATTR} concepts each)")


if __name__ == "__main__":
    main()
