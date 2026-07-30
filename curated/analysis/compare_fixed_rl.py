#!/usr/bin/env python3
"""Paired standard-versus-RLv2 comparison on identical valid swap images.

The primary causal quantity is response_delta:
  (z_donor - z_source)_counterfactual - (z_donor - z_source)_original.
Absolute post-swap ordering remains descriptive because a constant/blank image
can retain concept-coordinate rankings without responding to the intervention.
"""
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
    required = set(keys + [
        "part", "direction", "margin", "ordering_correct",
        "response_delta", "swap_moved_toward_donor",
    ])
    for name, frame, path in [
        ("standard", standard, standard_path),
        ("relabeled", relabeled, rl_path),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} ({name}) missing columns: {sorted(missing)}")
        if frame[keys].duplicated().any():
            raise RuntimeError(f"{path.name} has duplicate fixed-image keys")

    keep = keys + [
        "part", "direction", "margin", "ordering_correct",
        "response_delta", "swap_moved_toward_donor",
    ]
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
    merged["response_positive_standard"] = as_bool(
        merged["swap_moved_toward_donor_standard"]
    )
    merged["response_positive_rl"] = as_bool(
        merged["swap_moved_toward_donor_rl"]
    )
    merged["cross_model_delta_margin"] = merged["margin_rl"] - merged["margin_standard"]
    merged["cross_model_delta_response"] = (
        merged["response_delta_rl"] - merged["response_delta_standard"]
    )
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
                "mean_response_standard": q["response_delta_standard"].mean(),
                "mean_response_rl": q["response_delta_rl"].mean(),
                "delta_mean_response": q["cross_model_delta_response"].mean(),
                "response_positive_standard": q["response_positive_standard"].mean(),
                "response_positive_rl": q["response_positive_rl"].mean(),
                "delta_response_positive": (
                    q["response_positive_rl"].mean() -
                    q["response_positive_standard"].mean()
                ),
                "mean_cross_model_delta_margin": q["cross_model_delta_margin"].mean(),
                "median_cross_model_delta_margin": q["cross_model_delta_margin"].median(),
                "fail_to_success_n": int(q["fail_to_success"].sum()),
                "success_to_fail_n": int(q["success_to_fail"].sum()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rl-tag", default="rlv2")
    args = ap.parse_args()
    out = Path(args.out)
    rows = []
    tag = re.escape(args.rl_tag)

    for rl_path in sorted(out.glob(f"funnybirds-cbm-{args.rl_tag}-s*.csv")):
        match = re.fullmatch(rf"funnybirds-cbm-{tag}-s(\d+)\.csv", rl_path.name)
        if not match:
            continue
        seed = int(match.group(1))
        standard_path = out / f"funnybirds-cbm-s{seed}.csv"
        if standard_path.exists():
            rows.extend(paired_summary(standard_path, rl_path, "CBM", np.nan, seed))

    for rl_path in sorted(out.glob(f"funnybirds-mcbm-{args.rl_tag}-g*-s*.csv")):
        match = re.fullmatch(
            rf"funnybirds-mcbm-{tag}-g([0-9p]+)-s(\d+)\.csv", rl_path.name
        )
        if not match:
            continue
        tag, seed_text = match.groups()
        seed = int(seed_text)
        gamma = float(tag.replace("p", "."))
        standard_path = out / f"funnybirds-mcbm-g{tag}-s{seed}.csv"
        if standard_path.exists():
            rows.extend(paired_summary(standard_path, rl_path, "MCBM", gamma, seed))

    if not rows:
        raise RuntimeError(
            f"no matched standard/{args.rl_tag} fixed CSV pairs found in {out}"
        )

    summary = pd.DataFrame(rows).sort_values(
        ["model", "gamma", "seed", "direction", "part"],
        na_position="first",
    )
    summary_path = (
        out / "fixed_rl_comparison.csv"
        if args.rl_tag == "rlv2"
        else out / f"fixed_rl_comparison_{args.rl_tag}.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n===== FIXED-IMAGE PAIRED RL COMPARISON (direction=all) =====")
    cols = [
        "model", "gamma", "seed", "part", "n",
        "mean_response_standard", "mean_response_rl", "delta_mean_response",
        "response_positive_standard", "response_positive_rl",
        "delta_response_positive",
        "ordering_standard", "ordering_rl", "delta_ordering",
        "mean_cross_model_delta_margin", "median_cross_model_delta_margin",
        "fail_to_success_n", "success_to_fail_n",
    ]
    print(
        summary.loc[summary["direction"] == "all", cols]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
