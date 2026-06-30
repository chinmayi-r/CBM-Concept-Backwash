"""
curated/data/cub/build_cub_cbm_data.py

Build the pickled list-of-dicts expected by the official CBM trainer
(yewsiang/ConceptBottleneck) from the standard CUB-200-2011 dataset.
Matches CUB/data_processing.py and CUB/dataset.py from the official repo:

  * Three splits: train/val/test. test = official CUB train_test_split.txt
    test images; val = a random 20% of the official train images (held out,
    seeded); train = the remaining 80%. (data_processing.py, val_ratio=0.2)
  * img_path must contain "CUB_200_2011" as a path component — dataset.py
    locates this token and rebuilds the path relative to image_dir at load
    time, so cub_root's directory name MUST be CUB_200_2011.
  * Record schema: id, img_path, class_label, attribute_label,
    attribute_certainty (dataset.py reads img_path, class_label,
    attribute_label, attribute_certainty / uncertain_attribute_label).

The 112 binary attributes used in the CBM paper (and reused verbatim by
minimal_cbm's CUB200 loader, USED_ATTRIBUTES) are a hand-selected subset of
CUB's 312 attributes. These pkls are shared by both the CBM and MCBM training
scripts — MCBM's CUB200 dataset reads the same train.pkl/test.pkl schema.

Outputs (to $CURATED_DATA/CUB_processed/class_attr_data_10/):
    train.pkl, val.pkl, test.pkl

Usage:
    python build_cub_cbm_data.py \\
        --cub_root /path/to/CUB_200_2011 \\
        --out_dir  $CURATED_DATA/CUB_processed/class_attr_data_10 \\
        [--val_ratio 0.2] [--seed 42]
"""

from __future__ import annotations
import argparse
import pickle
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.cub_metadata import load_cub_metadata

# 112 attribute IDs used in Koh et al. 2020 (CBM paper), 1-indexed.
# Source: ConceptBottleneck/CUB/data_utils.py SELECTED_CONCEPTS, also reused
# verbatim as USED_ATTRIBUTES in minimal_cbm/src/datasets/cub200.py.
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


def _build_records(cub_root: Path, img_ids: list[int], meta) -> list[dict]:
    images = meta.images.set_index("image_id")
    labels = meta.image_class_labels.set_index("image_id")
    iab = meta.image_attributes_binary.set_index("image_id")
    attr_cols = [f"attr_{aid}_present" for aid in SELECTED_ATTR_IDS]

    records = []
    for img_id in img_ids:
        cls_id = int(labels.loc[img_id, "class_id"]) - 1  # 0-based
        rel_path = images.loc[img_id, "file_path"]
        # Keep cub_root's own name (must be "CUB_200_2011") in the path so the
        # official dataset.py can find the "CUB_200_2011" token and rebuild
        # the path relative to its own image_dir at load time.
        img_path = cub_root / "images" / rel_path
        attr_vec = [int(iab.loc[img_id, c]) if img_id in iab.index else 0
                    for c in attr_cols]
        records.append({
            "id":                   str(img_id),
            "img_path":             str(img_path.resolve()),
            "class_label":          cls_id,
            "attribute_label":      attr_vec,
            "attribute_certainty":  [1.0] * N_ATTR,
        })
    return records


def build_splits(cub_root: Path, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    if cub_root.name != "CUB_200_2011":
        raise ValueError(
            f"cub_root must be a directory literally named 'CUB_200_2011' "
            f"(got {cub_root.name!r}) — the official dataset.py path "
            f"reconstruction depends on finding that token in img_path."
        )

    meta = load_cub_metadata(cub_root)
    if meta.image_attributes_binary is None:
        raise RuntimeError(
            "CUB attribute annotations not found. "
            "Ensure image_attribute_labels.txt is present under cub_root."
        )

    splits = meta.train_test_split.set_index("image_id")
    is_train = splits["is_training_image"] == 1
    train_val_ids = splits.index[is_train].tolist()
    test_ids = splits.index[~is_train].tolist()

    rng = random.Random(seed)
    shuffled = list(train_val_ids)
    rng.shuffle(shuffled)
    n_val = int(val_ratio * len(shuffled))
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]

    return {
        "train": _build_records(cub_root, train_ids, meta),
        "val":   _build_records(cub_root, val_ids, meta),
        "test":  _build_records(cub_root, test_ids, meta),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub_root",  required=True, type=Path)
    ap.add_argument("--out_dir",   required=True, type=Path)
    ap.add_argument("--val_ratio", default=0.2, type=float)
    ap.add_argument("--seed",      default=42, type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = build_splits(args.cub_root, args.val_ratio, args.seed)
    for split, records in splits.items():
        out = args.out_dir / f"{split}.pkl"
        with open(out, "wb") as f:
            pickle.dump(records, f)
        print(f"  → {out}  ({len(records)} records, {N_ATTR} concepts each)")


if __name__ == "__main__":
    main()
