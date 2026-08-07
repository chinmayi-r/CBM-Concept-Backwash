#!/usr/bin/env python3
"""Create read-only Koh split views from the already accepted FunnyBird records.

No labels or records are regenerated. The views contain symlinks only:
train -> existing train-selection split, val -> its existing validation split,
test -> the existing untouched test split.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def load(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def identity(row):
    return str(row.get("image", row.get("img_path"))), int(row["class_label"])


def same(left, right) -> bool:
    try:
        return bool(np.array_equal(left, right))
    except (TypeError, ValueError):
        return left == right


def compare(left_path: Path, right_path: Path, split: str) -> dict:
    left, right = load(left_path), load(right_path)
    if [identity(row) for row in left] != [identity(row) for row in right]:
        raise RuntimeError(f"standard/RLv2 image or class mismatch in {split}")
    changed = 0
    for standard, rlv2 in zip(left, right):
        keys = set(standard) | set(rlv2)
        illegal = [key for key in keys - {"attribute_label"}
                   if not same(standard.get(key), rlv2.get(key))]
        if illegal:
            raise RuntimeError(
                f"RLv2 changed non-label fields in {split}: "
                f"image={identity(standard)} fields={illegal}"
            )
        changed += not same(standard.get("attribute_label"),
                            rlv2.get("attribute_label"))
    return {"rows": len(left), "rows_with_changed_attribute_label": changed}


def link(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"existing link points to wrong source: {target}")
        return
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing path: {target}")
    target.symlink_to(source.resolve())


def write_funnybird_path_view(source: Path, target: Path, image_root: Path) -> None:
    """Copy records while changing only img_path to Koh's expected marker path."""
    records = load(source)
    converted = []
    for row in records:
        updated = copy.copy(row)
        original = Path(str(row["img_path"])).resolve()
        try:
            relative = original.relative_to(image_root)
        except ValueError as exc:
            raise RuntimeError(
                f"FunnyBird image is outside FUNNYBIRDS_ROOT: {original}"
            ) from exc
        updated["img_path"] = (Path("CUB_200_2011") / relative).as_posix()
        converted.append(updated)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(converted, stream)
    temporary.replace(target)
    # Prove that the view changed paths only.
    reread = load(target)
    for before, after in zip(records, reread):
        if set(before) != set(after):
            raise RuntimeError("path view changed record fields")
        for key in before:
            if key != "img_path" and not same(before[key], after[key]):
                raise RuntimeError(f"path view changed non-path field: {key}")
        if not str(after["img_path"]).startswith("CUB_200_2011/"):
            raise RuntimeError("path view lacks Koh CUB_200_2011 marker")


def link_dir(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing image view: {target}")
    target.symlink_to(source.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    parser.add_argument("--funnybirds-root", default=os.environ.get("FUNNYBIRDS_ROOT", ""))
    parser.add_argument("--cub-root", default=os.environ.get("CUB_ROOT", ""))
    args = parser.parse_args()
    if not args.data_root or not args.funnybirds_root or not args.cub_root:
        parser.error("set CURATED_DATA, FUNNYBIRDS_ROOT, and CUB_ROOT")
    root = Path(args.data_root).resolve()

    sources = {
        "standard": {
            "train": root / "funnybirds_processed_trainval/train.pkl",
            "val": root / "funnybirds_processed_trainval/test.pkl",
            "test": root / "funnybirds_processed/test.pkl",
        },
        "rlv2": {
            "train": root / "funnybirds_processed_rl_trainval/train.pkl",
            "val": root / "funnybirds_processed_rl_trainval/test.pkl",
            "test": root / "funnybirds_processed_rl/test.pkl",
        },
    }
    parity = {
        split: compare(sources["standard"][split], sources["rlv2"][split], split)
        for split in ("train", "val", "test")
    }
    image_root = Path(args.funnybirds_root).resolve()
    out = root / "koh_joint_inputs/funnybirds"
    for labels, splits in sources.items():
        for split, source in splits.items():
            write_funnybird_path_view(
                source, out / labels / f"{split}.pkl", image_root
            )
    manifest = {
        "status": "SUCCESS",
        "operation": (
            "copy existing accepted records and rewrite img_path only to the "
            "CUB_200_2011 marker expected by Koh's unchanged loader"
        ),
        "parity": parity,
        "sources": {
            labels: {split: str(path.resolve()) for split, path in splits.items()}
            for labels, splits in sources.items()
        },
    }
    path = out / "INPUTS.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    views = root / "koh_joint_inputs/work"
    link_dir(Path(args.funnybirds_root).resolve(),
             views / "funnybirds/CUB_200_2011")
    link_dir(Path(args.cub_root).resolve(), views / "cub70/CUB_200_2011")
    link_dir(Path(args.cub_root).resolve(), views / "cub/CUB_200_2011")

    curated = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(curated / "data/funnybirds"))
    import funnybirds_concepts
    parts = funnybirds_concepts.load_parts(Path(args.funnybirds_root).resolve())
    (root / "koh_joint_inputs/funnybird_concept_names.json").write_text(
        json.dumps(funnybirds_concepts.concept_names(parts), indent=2) + "\n"
    )
    print(f"[KOH INPUT VIEW PASS] {path}")


if __name__ == "__main__":
    main()
