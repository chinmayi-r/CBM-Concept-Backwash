#!/usr/bin/env python3
"""Derive corrected FunnyBird target-part areas without changing accepted swaps.

The original fixed-swap CSV counted only the first renderer instance for eye,
wing, and foot.  This read-only derivation sums both official instance colors,
checks the stored part-map hashes, and writes a keyed correction table plus an
auditable manifest.  It performs no model inference or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


COLORS = {
    "beak": ((255, 255, 0),),
    "eye": ((255, 255, 253), (255, 255, 254)),
    "wing": ((0, 255, 1), (0, 255, 2)),
    "foot": ((255, 0, 1), (255, 0, 2)),
    "tail": ((0, 0, 255),),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count(arr: np.ndarray, colors: tuple[tuple[int, int, int], ...]) -> int:
    return int(sum(np.all(arr == np.asarray(color), axis=2).sum() for color in colors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    csvs = [
        args.swap_root / "funnybirds-cbm-s1.csv",
        args.swap_root / "funnybirds-cbm-rlv2matched-s1.csv",
    ]
    frames = []
    for csv_path in csvs:
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(csv_path)
        required = {"render_id", "part", "image_cf_path", "partmap_cf_sha256", "pixel_count_cf"}
        missing = required - set(frame)
        if missing:
            raise RuntimeError(f"{csv_path.name} lacks {sorted(missing)}")
        frame = frame[list(required)].copy()
        frame["source_csv"] = csv_path.name
        frames.append(frame)

    both = pd.concat(frames, ignore_index=True)
    identity = ["render_id", "part"]
    disagreement = both.groupby(identity).agg(
        hashes=("partmap_cf_sha256", "nunique"),
        paths=("image_cf_path", "nunique"),
        legacy_counts=("pixel_count_cf", "nunique"),
    )
    if (disagreement > 1).any(axis=None):
        raise RuntimeError("standard and RLv2 disagree on a fixed-render identity")

    unique = both.drop_duplicates(identity).sort_values(identity).reset_index(drop=True)
    rows = []
    for row in unique.itertuples(index=False):
        rgb_path = Path(row.image_cf_path)
        partmap = rgb_path.parent.parent / "part_map" / f"{row.render_id}.png"
        if not partmap.is_file():
            raise FileNotFoundError(partmap)
        actual_hash = sha256(partmap)
        if actual_hash != str(row.partmap_cf_sha256):
            raise RuntimeError(f"part-map hash mismatch for {partmap}")
        with Image.open(partmap) as image:
            arr = np.asarray(image.convert("RGB"))
        colors = COLORS[str(row.part)]
        first = count(arr, colors[:1])
        corrected = count(arr, colors)
        legacy = int(row.pixel_count_cf)
        if first != legacy:
            raise RuntimeError(
                f"legacy count mismatch for {row.render_id}/{row.part}: "
                f"stored={legacy}, recomputed={first}"
            )
        rows.append({
            "render_id": str(row.render_id),
            "part": str(row.part),
            "legacy_single_instance_pixels": legacy,
            "corrected_all_instance_pixels": corrected,
            "added_second_instance_pixels": corrected - legacy,
            "partmap_cf_sha256": actual_hash,
        })

    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table_path = args.output / "visibility.csv"
    result.to_csv(table_path, index=False)
    summary = result.groupby("part").agg(
        rows=("render_id", "size"),
        mean_legacy_pixels=("legacy_single_instance_pixels", "mean"),
        mean_corrected_pixels=("corrected_all_instance_pixels", "mean"),
        rows_changed=("added_second_instance_pixels", lambda x: int((x > 0).sum())),
    ).reset_index()
    summary.to_csv(args.output / "summary_by_part.csv", index=False)
    manifest = {
        "status": "SUCCESS",
        "derivation": "sum_all_official_renderer_instance_colors_v1",
        "source_csvs": [str(path) for path in csvs],
        "rows": len(result),
        "key": ["render_id", "part"],
        "table": str(table_path),
        "table_sha256": sha256(table_path),
    }
    (args.output / "SUCCESS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"[VISIBILITY CORRECTION SUCCESS] {args.output / 'SUCCESS.json'}")


if __name__ == "__main__":
    main()
