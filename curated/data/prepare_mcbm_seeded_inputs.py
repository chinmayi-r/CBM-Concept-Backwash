#!/usr/bin/env python3
"""Create MCBM model-selection/final-test views from existing split files.

No records are regenerated. minimal_cbm calls its evaluation split `test.pkl`,
so the `selection` view deliberately points that name at validation. The real
test remains separate under `final/test.pkl`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing input view: {target}")
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set CURATED_DATA or pass --data-root")
    root = Path(args.data_root).resolve()
    sources = {
        ("funnybirds", "standard"): {
            "train": root / "funnybirds_processed_trainval/train.pkl",
            "val": root / "funnybirds_processed_trainval/test.pkl",
            "test": root / "funnybirds_processed/test.pkl",
        },
        ("funnybirds", "rlv2"): {
            "train": root / "funnybirds_processed_rl_trainval/train.pkl",
            "val": root / "funnybirds_processed_rl_trainval/test.pkl",
            "test": root / "funnybirds_processed_rl/test.pkl",
        },
        ("cub70", "standard"): {
            "train": root / "CUB_processed/class_attr_data_10_cub70_original/train.pkl",
            "val": root / "CUB_processed/class_attr_data_10_cub70_original/val.pkl",
            "test": root / "CUB_processed/class_attr_data_10_cub70_original/test.pkl",
        },
        ("cub", "standard"): {
            "train": root / "CUB_processed/class_attr_data_10/train.pkl",
            "val": root / "CUB_processed/class_attr_data_10/val.pkl",
            "test": root / "CUB_processed/class_attr_data_10/test.pkl",
        },
    }
    out = root / "mcbm_seeded_v1_inputs"
    manifest = {"status": "SUCCESS", "operation": "symlink existing records"}
    for (dataset, labels), split in sources.items():
        base = out / dataset / labels
        link(split["train"], base / "selection/train.pkl")
        link(split["val"], base / "selection/test.pkl")
        link(split["test"], base / "final/test.pkl")
        manifest[f"{dataset}:{labels}"] = {
            name: str(path.resolve()) for name, path in split.items()
        }
    (out / "INPUTS.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"[MCBM INPUT VIEW PASS] {out / 'INPUTS.json'}")


if __name__ == "__main__":
    main()
