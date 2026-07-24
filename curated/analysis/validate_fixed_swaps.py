#!/usr/bin/env python3
"""Fail closed unless fixed-render swap CSVs reused identical image bytes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    files = sorted(
        p for p in out.glob("*-s*.csv")
        if p.stem.rsplit("-s", 1)[-1].isdigit()
    )
    if not files:
        raise RuntimeError(f"no combined swap CSVs found in {out}")

    required = {
        "li", "render_id", "image_cf_sha256", "partmap_cf_sha256",
        "orig_render_id", "image_orig_sha256", "part", "direction",
        "sid_src", "sid_donor", "var_src", "var_donor",
    }
    render_hashes: dict[str, set[str]] = {}
    partmap_hashes: dict[str, set[str]] = {}
    original_hashes: dict[str, set[str]] = {}
    expected_ids: set[str] | None = None
    expected_from = ""

    for path in files:
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{path.name} lacks fixed-render columns: {sorted(missing)}")
        if df["render_id"].duplicated().any():
            examples = df.loc[df["render_id"].duplicated(), "render_id"].head().tolist()
            raise RuntimeError(f"{path.name} has duplicate render IDs: {examples}")
        ids = set(df["render_id"].astype(str))
        if expected_ids is None:
            expected_ids, expected_from = ids, path.name
        elif ids != expected_ids:
            raise RuntimeError(
                f"{path.name} does not evaluate the same render IDs as {expected_from}: "
                f"missing={len(expected_ids - ids)}, extra={len(ids - expected_ids)}"
            )
        for rid, sha in zip(df["render_id"], df["image_cf_sha256"]):
            render_hashes.setdefault(str(rid), set()).add(str(sha))
        for rid, sha in zip(df["render_id"], df["partmap_cf_sha256"]):
            if pd.notna(sha) and str(sha):
                partmap_hashes.setdefault(str(rid), set()).add(str(sha))
        for rid, sha in zip(df["orig_render_id"], df["image_orig_sha256"]):
            original_hashes.setdefault(str(rid), set()).add(str(sha))
        print(f"[checked] {path.name}: {len(df)} unique fixed renders")

    bad_rgb = {rid: hs for rid, hs in render_hashes.items() if len(hs) != 1}
    bad_seg = {rid: hs for rid, hs in partmap_hashes.items() if len(hs) != 1}
    bad_orig = {rid: hs for rid, hs in original_hashes.items() if len(hs) != 1}
    if bad_rgb or bad_seg or bad_orig:
        raise RuntimeError(
            "fixed-render hash mismatch: "
            f"counterfactual_rgb={len(bad_rgb)}, part_map={len(bad_seg)}, "
            f"original_rgb={len(bad_orig)}"
        )

    print(
        "FIXED SWAP VALIDATION PASSED: "
        f"{len(files)} model CSVs, {len(render_hashes)} counterfactual RGB IDs, "
        f"{len(original_hashes)} original RGB IDs; every reused ID has one SHA-256."
    )


if __name__ == "__main__":
    main()
