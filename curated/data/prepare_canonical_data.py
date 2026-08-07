#!/usr/bin/env python3
"""Build immutable train/validation/test inputs for the canonical rebuild.

This script intentionally creates a new namespace.  Koh receives real
``train.pkl``, ``val.pkl`` and untouched ``test.pkl`` files.  minimal_cbm
receives ``train.pkl`` plus a ``test.pkl`` that is *validation* during model
selection; its untouched final test file is stored separately and recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def dump(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_train(records, seed=42, fraction=0.1):
    by_class = defaultdict(list)
    for row in records:
        by_class[int(row["class_label"])].append(row)
    rng = random.Random(seed)
    train, val = [], []
    for class_id in sorted(by_class):
        rows = list(by_class[class_id])
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * fraction))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def identity(row):
    return str(row.get("image", row["img_path"])), int(row["class_label"])


def assert_matched(standard, relabeled, split):
    a = [identity(row) for row in standard]
    b = [identity(row) for row in relabeled]
    if a != b:
        raise RuntimeError(f"standard/RLv2 {split} image or class populations differ")
    illegal = []
    for left, right in zip(standard, relabeled):
        keys = set(left) | set(right)
        for key in keys - {"attribute_label"}:
            if left.get(key) != right.get(key):
                illegal.append((identity(left), key))
                break
    if illegal:
        raise RuntimeError(f"RLv2 changed fields other than attribute_label: {illegal[:3]}")


def write_layout(raw: Path, target: Path, *, rewrite_root: Path | None = None):
    source_train, source_test = load(raw / "train.pkl"), load(raw / "test.pkl")
    train, val = split_train(source_train)
    if rewrite_root is not None:
        for rows in (train, val, source_test):
            for row in rows:
                rel = row.get("image")
                if not rel:
                    raise RuntimeError("FunnyBird row lacks stable image identity")
                row["img_path"] = str(rewrite_root / rel)
    koh = target / "koh"
    mcbm = target / "mcbm_selection"
    final = target / "final_test"
    for name, rows in (("train", train), ("val", val), ("test", source_test)):
        dump(rows, koh / f"{name}.pkl")
    dump(train, mcbm / "train.pkl")
    dump(val, mcbm / "test.pkl")
    dump(source_test, final / "test.pkl")
    return {"train": len(train), "val": len(val), "test": len(source_test)}


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--funnybirds-root", required=True, type=Path)
    ap.add_argument("--cub-root", required=True, type=Path)
    ap.add_argument("--cub-attributes", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    curated = repo / "curated"
    root = args.data_root.resolve()
    marker = root / "canonical_data_manifest.json"
    if root.exists():
        if not args.force:
            raise SystemExit(f"canonical data root already exists: {root}; use a new root")
        resolved = root.resolve()
        if resolved == Path("/") or len(resolved.parts) < 4:
            raise SystemExit(f"refusing broad deletion: {resolved}")
        shutil.rmtree(resolved)
    root.mkdir(parents=True)

    raw = root / "raw_build"
    run([
        "python3", curated / "data/funnybirds/build_funnybirds_cbm_data.py",
        "--data-root", raw, "--funnybirds-root", args.funnybirds_root,
        "--labels", "species", "--out-name", "funnybirds_standard",
    ])
    run([
        "python3", curated / "data/funnybirds/build_funnybirds_cbm_data.py",
        "--data-root", raw, "--funnybirds-root", args.funnybirds_root,
        "--labels", "image_level", "--out-name", "funnybirds_rlv2",
    ])

    # Put CUB_200_2011 in the path so the unmodified Koh loader follows its
    # normal path-resolution branch.  The link points only to immutable raw RGB.
    fb_link = root / "funnybirds_images" / "CUB_200_2011"
    fb_link.parent.mkdir(parents=True)
    os.symlink(args.funnybirds_root.resolve(), fb_link, target_is_directory=True)
    for dataset, source_root in (
        ("funnybirds", args.funnybirds_root.resolve()),
        ("cub", args.cub_root.resolve()),
        ("cub70", args.cub_root.resolve()),
    ):
        work_link = root / "koh_work" / dataset / "CUB_200_2011"
        work_link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source_root, work_link, target_is_directory=True)
    counts = {}
    counts["funnybirds_standard"] = write_layout(
        raw / "funnybirds_standard", root / "funnybirds_standard",
        rewrite_root=fb_link,
    )
    counts["funnybirds_rlv2"] = write_layout(
        raw / "funnybirds_rlv2", root / "funnybirds_rlv2",
        rewrite_root=fb_link,
    )
    sys.path.insert(0, str(curated / "data/funnybirds"))
    import funnybirds_concepts
    parts = funnybirds_concepts.load_parts(args.funnybirds_root)
    (root / "funnybirds_concept_names.json").write_text(
        json.dumps(funnybirds_concepts.concept_names(parts), indent=2) + "\n"
    )
    for split in ("train", "val", "test"):
        assert_matched(
            load(root / f"funnybirds_standard/koh/{split}.pkl"),
            load(root / f"funnybirds_rlv2/koh/{split}.pkl"), split,
        )

    cub_build = raw / "cub"
    run([
        "python3", curated / "data/cub/prepare_cub_data.py",
        "--cub-root", args.cub_root, "--data-root", cub_build,
        "--attr-names", args.cub_attributes, "--force",
    ])
    for dataset, source in (
        ("cub", cub_build / "CUB_processed/class_attr_data_10"),
        ("cub70", cub_build / "CUB_processed/class_attr_data_10_cub70_original"),
    ):
        target = root / dataset
        koh = target / "koh"
        koh.mkdir(parents=True)
        for split in ("train", "val", "test"):
            shutil.copy2(source / f"{split}.pkl", koh / f"{split}.pkl")
        shutil.copy2(source / "selection_indices.json", target / "selection_indices.json")
        shutil.copy2(cub_build / "CUB_processed/attributes.txt", target / "attributes.txt")
        train, val, test = (load(koh / f"{s}.pkl") for s in ("train", "val", "test"))
        dump(train, target / "mcbm_selection/train.pkl")
        dump(val, target / "mcbm_selection/test.pkl")
        dump(test, target / "final_test/test.pkl")
        counts[dataset] = {"train": len(train), "val": len(val), "test": len(test)}

    files = sorted(root.rglob("*.pkl"))
    manifest = {
        "status": "SUCCESS", "schema": 1, "split_seed": 42,
        "counts": counts,
        "files": {str(path.relative_to(root)): digest(path) for path in files},
        "funnybirds_root": str(args.funnybirds_root.resolve()),
        "cub_root": str(args.cub_root.resolve()),
        "cub_attributes": str(args.cub_attributes.resolve()),
    }
    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[CANONICAL DATA SUCCESS] {marker}")


if __name__ == "__main__":
    main()
