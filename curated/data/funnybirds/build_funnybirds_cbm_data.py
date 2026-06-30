"""
curated/data/funnybirds/build_funnybirds_cbm_data.py

Build the pickled list-of-dicts expected by the official CBM trainer
(yewsiang/ConceptBottleneck) from the raw FunnyBirds dataset.

KNOWN COMPATIBILITY GAP (read before running):
  CUB/dataset.py's image loader is hardcoded to CUB conventions: when
  `image_dir != 'images'`, it does
      idx = img_path.split('/').index('CUB_200_2011')
  to find where to splice in `image_dir`. FunnyBirds paths have no such
  token, so this throws ValueError on the unmodified official loader.
  Per curated/README.md, nothing in external/ may be edited, so this builder
  works around it the same way the official repo expects CUB_processed to be
  laid out: it creates a symlink named CUB_200_2011 pointing at
  `funnybirds_root` inside `out_dir`, and writes img_path through that
  symlink. Pass `-image_dir <out_dir>/CUB_200_2011` to experiments.py (see
  curated/train/cbm_funnybirds.sh) so the token search succeeds.
  This is a path-construction shim only — no FunnyBirds-specific behavior is
  added to the official trainer.

Record schema (matches CUB/dataset.py):
    id, img_path, class_label, attribute_label, attribute_certainty

Splits: FunnyBirds ships dataset_train.json / dataset_test.json (2-way) with
no official val split. We carve 20% off train the same way
CUB/data_processing.py does (val_ratio=0.2), so the official trainer's
expectation of train/val/test all being present is met consistently across
both datasets.

Outputs (to $CURATED_DATA/funnybirds_processed/):
    train.pkl, val.pkl, test.pkl, CUB_200_2011 (symlink -> funnybirds_root)

Usage:
    python build_funnybirds_cbm_data.py \\
        --funnybirds_root /path/to/FunnyBirds \\
        --out_dir         $CURATED_DATA/funnybirds_processed \\
        [--concept_labels /path/to/concept_labels_image_level.npy] \\
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

from datasets.funnybirds_dataset import FunnyBirdsDataset, NUM_CONCEPTS


def _make_records(funnybirds_root: Path, symlink_root: Path, split: str,
                   indices: list[int], concept_labels_path: Path | None) -> list[dict]:
    ds = FunnyBirdsDataset(
        funnybirds_root=funnybirds_root,
        split=split,
        transform=None,
        include_concepts=True,
        concept_labels_path=concept_labels_path if split == "train" else None,
    )
    records = []
    for idx in indices:
        sample = ds[idx]
        class_idx = int(sample["label"].item())
        # Path goes through the CUB_200_2011 symlink so the official
        # CUB/dataset.py path-token search succeeds unmodified.
        img_path = symlink_root / split / str(class_idx) / f"{idx:06d}.png"
        concepts = sample["concepts"].tolist()
        records.append({
            "id":                   f"{split}_{idx:06d}",
            "img_path":             str(img_path),
            "class_label":          class_idx,
            "attribute_label":      [int(c) for c in concepts],
            "attribute_certainty":  [1.0] * NUM_CONCEPTS,
        })
    return records


def build_splits(funnybirds_root: Path, symlink_root: Path, val_ratio: float,
                  seed: int, concept_labels_path: Path | None) -> dict[str, list[dict]]:
    train_full = FunnyBirdsDataset(funnybirds_root, split="train")
    test_full = FunnyBirdsDataset(funnybirds_root, split="test")

    rng = random.Random(seed)
    train_idx_all = list(range(len(train_full)))
    rng.shuffle(train_idx_all)
    n_val = int(val_ratio * len(train_idx_all))
    val_idx = sorted(train_idx_all[:n_val])
    train_idx = sorted(train_idx_all[n_val:])
    test_idx = list(range(len(test_full)))

    return {
        "train": _make_records(funnybirds_root, symlink_root, "train", train_idx, concept_labels_path),
        "val":   _make_records(funnybirds_root, symlink_root, "train", val_idx, concept_labels_path),
        "test":  _make_records(funnybirds_root, symlink_root, "test", test_idx, None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds_root", required=True, type=Path)
    ap.add_argument("--out_dir",         required=True, type=Path)
    ap.add_argument("--concept_labels",  default=None,  type=Path,
                    help="optional (N_train, 26) .npy from make_image_level_concept_labels.py")
    ap.add_argument("--val_ratio", default=0.2, type=float)
    ap.add_argument("--seed",      default=42, type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    symlink_root = args.out_dir / "CUB_200_2011"
    if not symlink_root.exists():
        symlink_root.symlink_to(args.funnybirds_root.resolve(), target_is_directory=True)
        print(f"  created {symlink_root} -> {args.funnybirds_root.resolve()}")

    print("Building splits…")
    splits = build_splits(args.funnybirds_root, symlink_root, args.val_ratio,
                           args.seed, args.concept_labels)
    for split, records in splits.items():
        out = args.out_dir / f"{split}.pkl"
        with open(out, "wb") as f:
            pickle.dump(records, f)
        print(f"  → {out}  ({len(records)} records, {NUM_CONCEPTS} concepts each)")


if __name__ == "__main__":
    main()
