#!/usr/bin/env python3
"""Paired standard-versus-RLv2 comparison on identical cached swap images."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise RuntimeError(f"cannot parse ordering_correct values: {series[mapped.isna()].unique()}")
    return mapped.astype(bool)


def paired_summary(standard_path: Path, rl_path: Path, model: str, gamma, seed: int):
    standard = pd.read_csv(standard_path)
    relabeled = pd.read_csv(rl_path)
    keys = ["render_id", "image_cf_sha256"]
    required = set(keys + ["part", "direction", "margin", "ordering_correct"])
    for name, frame, path in [
        ("standard", standard, standard_path),
        ("relabeled", relabeled, rl_path),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} ({name}) missing columns: {sorted(missing)}")
        if frame[keys].duplicated().any():
            raise RuntimeError(f"{path.name} has duplicate fixed-image keys")

    keep = keys + ["part", "direction", "margin", "ordering_correct"]
    for optional in ["li", "pixel_count_cf", "sid_src", "sid_donor", "var_src", "var_donor"]:
        if optional in standard.columns and optional in relabeled.columns:
            keep.append(optional)
    merged = standard[keep].merge(
        relabeled[keep],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_standard", "_rl"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise RuntimeError(
            f"{standard_path.name} versus {rl_path.name} do not contain identical "
            f"fixed images: {counts}"
        )
    merged = merged.drop(columns="_merge")

    for col in ["part", "direction", "li", "pixel_count_cf", "sid_src", "sid_donor",
                "var_src", "var_donor"]:
        left, right = f"{col}_standard", f"{col}_rl"
        if left in merged and right in merged:
            if not merged[left].equals(merged[right]):
                raise RuntimeError(f"paired rows disagree on {col}")
            merged[col] = merged[left]

    merged["ordering_standard"] = as_bool(merged["ordering_correct_standard"])
    merged["ordering_rl"] = as_bool(merged["ordering_correct_rl"])
    merged["delta_margin"] = merged["margin_rl"] - merged["margin_standard"]
    merged["fail_to_success"] = (~merged["ordering_standard"]) & merged["ordering_rl"]
    merged["success_to_fail"] = merged["ordering_standard"] & (~merged["ordering_rl"])

    rows = []
    groups = [("all", merged)]
    groups.extend((direction, q) for direction, q in merged.groupby("direction"))
    for direction, frame in groups:
        for part, q in frame.groupby("part"):
            rows.append({
                "model": model,
                "gamma": gamma,
                "seed": seed,
                "direction": direction,
                "part": part,
                "n": len(q),
                "ordering_standard": q["ordering_standard"].mean(),
                "ordering_rl": q["ordering_rl"].mean(),
                "delta_ordering": q["ordering_rl"].mean() - q["ordering_standard"].mean(),
                "mean_delta_margin": q["delta_margin"].mean(),
                "median_delta_margin": q["delta_margin"].median(),
                "frac_delta_margin_positive": (q["delta_margin"] > 0).mean(),
                "fail_to_success_n": int(q["fail_to_success"].sum()),
                "success_to_fail_n": int(q["success_to_fail"].sum()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    rows = []

    for rl_path in sorted(out.glob("funnybirds-cbm-rlv2-s*.csv")):
        match = re.fullmatch(r"funnybirds-cbm-rlv2-s(\d+)\.csv", rl_path.name)
        if not match:
            continue
        seed = int(match.group(1))
        standard_path = out / f"funnybirds-cbm-s{seed}.csv"
        if standard_path.exists():
            rows.extend(paired_summary(standard_path, rl_path, "CBM", np.nan, seed))

    for rl_path in sorted(out.glob("funnybirds-mcbm-rlv2-g*-s*.csv")):
        match = re.fullmatch(r"funnybirds-mcbm-rlv2-g([0-9p]+)-s(\d+)\.csv", rl_path.name)
        if not match:
            continue
        tag, seed_text = match.groups()
        seed = int(seed_text)
        gamma = float(tag.replace("p", "."))
        standard_path = out / f"funnybirds-mcbm-g{tag}-s{seed}.csv"
        if standard_path.exists():
            rows.extend(paired_summary(standard_path, rl_path, "MCBM", gamma, seed))

    if not rows:
        raise RuntimeError(f"no matched standard/RLv2 fixed CSV pairs found in {out}")

    summary = pd.DataFrame(rows).sort_values(
        ["model", "gamma", "seed", "direction", "part"],
        na_position="first",
    )
    summary_path = out / "fixed_rl_comparison.csv"
    summary.to_csv(summary_path, index=False)

    print("\n===== FIXED-IMAGE PAIRED RL COMPARISON (direction=all) =====")
    cols = [
        "model", "gamma", "seed", "part", "n",
        "ordering_standard", "ordering_rl", "delta_ordering",
        "mean_delta_margin", "median_delta_margin",
        "frac_delta_margin_positive", "fail_to_success_n", "success_to_fail_n",
    ]
    print(
        summary.loc[summary["direction"] == "all", cols]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
