"""
curated/data/funnybirds/build_funnybirds_cbm_data.py

Build the pickled list-of-dicts expected by the official CBM trainer
(yewsiang/ConceptBottleneck) from the raw FunnyBirds dataset.

The CBM trainer's CUB dataloader (data_utils.py) expects a list of dicts:
    {
        "id":              <str>,       # unique image identifier
        "img_path":        <str>,       # absolute path to the image
        "class_label":     <int>,       # 0-based species id
        "attribute_label": <list[int]>, # binary concept vector, len = N_CONCEPTS
        "img_cover":       <float>,     # ignored; set to 1.0
        "attribute_certainty": <list[float]>, # ignored; set to 1.0 per concept
    }

Outputs (to $CURATED_DATA/funnybirds_processed/):
    train.pkl, test.pkl  — one pickle per split

Usage:
    python build_funnybirds_cbm_data.py \\
        --funnybirds_root /path/to/FunnyBirds \\
        --out_dir         $CURATED_DATA/funnybirds_processed \\
        [--concept_labels /path/to/concept_labels_image_level.npy]
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.funnybirds_dataset import FunnyBirdsDataset, NUM_CONCEPTS


def build_split(funnybirds_root: Path, split: str, concept_labels_path: Path | None) -> list[dict]:
    ds = FunnyBirdsDataset(
        funnybirds_root=funnybirds_root,
        split=split,
        transform=None,
        include_concepts=True,
        concept_labels_path=concept_labels_path if split == "train" else None,
    )
    records = []
    for idx in range(len(ds)):
        sample = ds[idx]
        class_idx = int(sample["label"].item())
        img_path = funnybirds_root / split / str(class_idx) / f"{idx:06d}.png"
        concepts = sample["concepts"].tolist()
        records.append({
            "id":                   f"{split}_{idx:06d}",
            "img_path":             str(img_path.resolve()),
            "class_label":          class_idx,
            "attribute_label":      [int(c) for c in concepts],
            "img_cover":            1.0,
            "attribute_certainty":  [1.0] * NUM_CONCEPTS,
        })
        if idx % 5000 == 0:
            print(f"  {split}: {idx}/{len(ds)}", flush=True)
    return records


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
        records = build_split(args.funnybirds_root, split, args.concept_labels)
        out = args.out_dir / f"{split}.pkl"
        with open(out, "wb") as f:
            pickle.dump(records, f)
        print(f"  → {out}  ({len(records)} records, {NUM_CONCEPTS} concepts each)")


if __name__ == "__main__":
    main()
