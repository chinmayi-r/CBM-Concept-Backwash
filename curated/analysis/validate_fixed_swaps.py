#!/usr/bin/env python3
"""Fail closed unless fixed-render CSVs reused identical *valid* image bytes.

Cross-model hash equality is necessary but not sufficient. A renderer that
returns one black PNG for every request satisfies equality perfectly.
"""
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
        unique_cf = int(df["image_cf_sha256"].nunique(dropna=True))
        unique_orig = int(df["image_orig_sha256"].nunique(dropna=True))
        min_unique_cf = max(10, int(len(df) * 0.01))
        n_orig_ids = int(df["orig_render_id"].nunique())
        min_unique_orig = max(5, int(n_orig_ids * 0.05))
        if unique_cf < min_unique_cf:
            raise RuntimeError(
                f"{path.name} has degenerate counterfactual render diversity: "
                f"{unique_cf} unique RGB hashes for {len(df)} rows; need >= {min_unique_cf}"
            )
        if unique_orig < min_unique_orig:
            raise RuntimeError(
                f"{path.name} has degenerate original render diversity: "
                f"{unique_orig} unique RGB hashes for {n_orig_ids} original IDs; "
                f"need >= {min_unique_orig}"
            )
        changed_from_orig = (
            df["image_cf_sha256"].astype(str) != df["image_orig_sha256"].astype(str)
        )
        changed_fraction = float(changed_from_orig.mean())
        if changed_fraction < 0.10:
            raise RuntimeError(
                f"{path.name} counterfactual RGB is usually identical to original: "
                f"changed_fraction={changed_fraction:.4f} < 0.10"
            )
        if df["partmap_cf_sha256"].notna().any():
            unique_seg = int(df["partmap_cf_sha256"].nunique(dropna=True))
            if unique_seg < min_unique_cf:
                raise RuntimeError(
                    f"{path.name} has degenerate part-map diversity: "
                    f"{unique_seg} unique hashes; need >= {min_unique_cf}"
                )
        if "pixel_count_cf" in df:
            positive_part_fraction = float((df["pixel_count_cf"].fillna(0) > 0).mean())
            if positive_part_fraction < 0.10:
                raise RuntimeError(
                    f"{path.name} part maps rarely contain the requested part: "
                    f"positive_pixel_count_fraction={positive_part_fraction:.4f} < 0.10"
                )
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
        print(
            f"[checked] {path.name}: {len(df)} fixed renders; "
            f"unique_cf_hashes={unique_cf}; unique_orig_hashes={unique_orig}; "
            f"changed_from_orig={changed_fraction:.3f}"
        )

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
        f"{len(original_hashes)} original RGB IDs; hashes agree across models and "
        "each CSV passed diversity/intervention checks."
    )


if __name__ == "__main__":
    main()
