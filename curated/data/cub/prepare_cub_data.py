#!/usr/bin/env python3
"""Build the exact CUB pickle inputs used by the curated CBM/MCBM runs.

This replaces the fragile multi-command recipe in the upstream
ConceptBottleneck repository.  In particular, that repository's
``CUB/data_processing.py`` currently references an undefined ``val_files``.
We preserve its extraction logic, make the split deterministic, validate every
stage, create the standard 112 class-level concepts, and filter classes 0..69
for CUB70.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parent
CURATED = HERE.parents[1]
UPSTREAM = CURATED / "external" / "ConceptBottleneck"
sys.path.insert(0, str(UPSTREAM))


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def valid_split(path: Path, expected_attributes: int) -> bool:
    if not path.exists():
        return False
    records = load_pickle(path)
    return bool(records) and len(records[0]["attribute_label"]) == expected_attributes


def validate_records(records, split: str, expected_attributes: int) -> None:
    if not records:
        raise RuntimeError(f"{split}: no records")
    widths = {len(record["attribute_label"]) for record in records}
    if widths != {expected_attributes}:
        raise RuntimeError(
            f"{split}: expected {expected_attributes} attributes, got widths {sorted(widths)}"
        )
    classes = {int(record["class_label"]) for record in records}
    if min(classes) < 0 or max(classes) >= 200:
        raise RuntimeError(f"{split}: invalid class range {min(classes)}..{max(classes)}")


def build_raw_pickles(cub_root: Path, output: Path, force: bool) -> None:
    targets = [output / f"{split}.pkl" for split in ("train", "val", "test")]
    if not force and all(valid_split(path, 312) for path in targets):
        print("[skip] complete 312-attribute CUB pickles already exist")
        return

    from CUB import data_processing

    # Upstream bug fix without modifying the git submodule.  extract_data()
    # resolves this name as a module global.
    data_processing.val_files = None
    random.seed(0)
    data_processing.random.seed(0)
    train, val, test = data_processing.extract_data(str(cub_root))
    splits = {"train": train, "val": val, "test": test}

    expected_counts = {"train": 4796, "val": 1198, "test": 5794}
    actual_counts = {name: len(records) for name, records in splits.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"unexpected deterministic split sizes: {actual_counts}; "
            f"expected {expected_counts}"
        )
    for name, records in splits.items():
        validate_records(records, name, 312)
        save_pickle(output / f"{name}.pkl", records)
        print(f"[write] {name}: {len(records)} -> {output / (name + '.pkl')}")


def build_class_level_pickles(output: Path, force: bool) -> Path:
    class_level = output / "class_attr_data_10"
    targets = [class_level / f"{split}.pkl" for split in ("train", "val", "test")]
    if not force and all(valid_split(path, 112) for path in targets):
        print("[skip] complete standard 112-concept pickles already exist")
        return class_level

    from CUB.generate_new_data import get_class_attributes_data

    # The upstream function opens "train.pkl" from the current directory.
    # Keep that quirk contained here.
    with working_directory(output):
        get_class_attributes_data(
            min_class_count=10,
            out_dir="class_attr_data_10",
            modify_data_dir=".",
        )
    for split, path in zip(("train", "val", "test"), targets):
        records = load_pickle(path)
        validate_records(records, split, 112)
        print(f"[check] {split}: {len(records)} images x 112 concepts")
    return class_level


def build_cub70_pickles(class_level: Path, output: Path, force: bool) -> Path:
    cub70 = output / "class_attr_data_10_cub70_original"
    targets = [cub70 / f"{split}.pkl" for split in ("train", "val", "test")]
    if not force and all(valid_split(path, 112) for path in targets):
        print("[skip] complete CUB70-filtered pickles already exist")
        return cub70

    for split in ("train", "val", "test"):
        source = load_pickle(class_level / f"{split}.pkl")
        kept = [record for record in source if 0 <= int(record["class_label"]) < 70]
        classes = sorted({int(record["class_label"]) for record in kept})
        if classes != list(range(70)):
            raise RuntimeError(f"{split}: expected classes 0..69, got {classes}")
        save_pickle(cub70 / f"{split}.pkl", kept)
        print(f"[write] CUB70 {split}: {len(kept)}/{len(source)} images")
    return cub70


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cub-root",
        default=os.environ.get("CUB_ROOT", ""),
        help="raw CUB_200_2011 directory (or set CUB_ROOT)",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CURATED_DATA", ""),
        help="artifact root (or set CURATED_DATA)",
    )
    parser.add_argument(
        "--attr-names",
        default=os.environ.get("CUB_ATTR_FILE", ""),
        help="312-line attributes.txt; defaults to CUB root's parent",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.cub_root or not args.data_root:
        parser.error("set CUB_ROOT and CURATED_DATA, or pass --cub-root/--data-root")
    cub_root = Path(args.cub_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    attr_names = (
        Path(args.attr_names).expanduser().resolve()
        if args.attr_names
        else cub_root.parent / "attributes.txt"
    )

    required = {
        cub_root / "images": "CUB images directory",
        cub_root / "images.txt": "images.txt",
        cub_root / "image_class_labels.txt": "image_class_labels.txt",
        cub_root / "train_test_split.txt": "train_test_split.txt",
        cub_root / "attributes" / "image_attribute_labels.txt": "per-image attributes",
        attr_names: "312-line attribute-name dictionary",
    }
    for path, description in required.items():
        require(path, description)
    lines = [line for line in attr_names.read_text().splitlines() if line.strip()]
    if len(lines) != 312:
        raise RuntimeError(f"{attr_names}: expected 312 attribute names, got {len(lines)}")

    output = data_root / "CUB_processed"
    output.mkdir(parents=True, exist_ok=True)
    # Put the dictionary next to processed data so training has a stable,
    # portable attr_dir independent of the raw dataset's unusual layout.
    shutil.copyfile(attr_names, output / "attributes.txt")

    print(f"CUB_ROOT={cub_root}")
    print(f"CUB_ATTR_FILE={attr_names}")
    print(f"CUB_PROCESSED={output}")
    build_raw_pickles(cub_root, output, args.force)
    class_level = build_class_level_pickles(output, args.force)
    cub70 = build_cub70_pickles(class_level, output, args.force)

    print("\nSUCCESS")
    print(f"full CUB: {class_level}")
    print(f"CUB70:    {cub70}")
    print(f"attr dir: {output}")


if __name__ == "__main__":
    main()
