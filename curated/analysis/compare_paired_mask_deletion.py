#!/usr/bin/env python3
"""Compare the identical mask-deletion experiment on FunnyBirds and CUB70."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REQUIRED = {
    "dataset", "image", "image_index", "class_label", "concept_idx",
    "concept_name", "part", "z_original",
    "z_target_deleted", "z_control_deleted", "z_part_only", "target_delta_z",
    "control_delta_z", "target_minus_control_z", "context_minus_part_only_z",
}
PART_ORDER = ["tail", "wing", "beak", "leg", "eye"]


def load(path: str, expected: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    if set(frame.dataset.astype(str)) != {expected}:
        raise ValueError(f"{path}: expected dataset={expected}")
    frame = frame.copy()
    frame["part_common"] = frame.part.replace({"foot": "leg"})
    return frame


def species_omega2(values: np.ndarray, labels: np.ndarray) -> float:
    """Bias-corrected one-way species effect; zero when group count explains noise."""
    values = np.asarray(values, dtype=float)
    codes, _ = pd.factorize(labels)
    n, k = len(values), len(np.unique(codes))
    if k < 2 or n <= k:
        return np.nan
    grand = values.mean()
    ss_between = 0.0
    ss_within = 0.0
    for code in range(k):
        local = values[codes == code]
        ss_between += len(local) * float((local.mean() - grand) ** 2)
        ss_within += float(np.square(local - local.mean()).sum())
    ss_total = ss_between + ss_within
    if ss_total <= 0:
        return np.nan
    ms_within = ss_within / (n - k)
    omega = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    return float(np.clip(omega, 0, 1))


def sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def calibrate_funnybird(shared: pd.DataFrame, clean_path: str, out: Path) -> dict:
    """Check whether shared inpainting agrees with the clean renderer deletion."""
    clean = pd.read_parquet(clean_path).copy()
    required = {"image_idx", "part", "typ_concept", "p_intact", "p_removed",
                "changed_frac"}
    missing = required - set(clean)
    if missing:
        raise ValueError(f"{clean_path} lacks calibration columns {sorted(missing)}")
    clean = clean[clean.changed_frac > 1e-3].copy()
    clean["clean_drop"] = clean.p_intact - clean.p_removed
    shared = shared.copy()
    shared["shared_drop"] = sigmoid(shared.z_original) - sigmoid(shared.z_target_deleted)
    merged = shared.merge(
        clean[["image_idx", "part", "typ_concept", "clean_drop"]],
        left_on=["image_index", "part", "concept_idx"],
        right_on=["image_idx", "part", "typ_concept"], how="inner",
        validate="many_to_one",
    )
    per = (merged.groupby("part")[["clean_drop", "shared_drop"]]
           .median().reset_index())
    per.to_csv(out / "funnybird_deletion_calibration.csv", index=False)
    row_rho = float(merged[["clean_drop", "shared_drop"]].corr(method="spearman").iloc[0, 1])
    part_rho = float(per[["clean_drop", "shared_drop"]].corr(method="spearman").iloc[0, 1])
    clean_positive = int((per.clean_drop > 0).sum())
    shared_positive = int((per.shared_drop > 0).sum())
    passed = (len(merged) >= 100 and clean_positive >= 4 and shared_positive >= 4 and
              np.isfinite(row_rho) and row_rho > 0 and np.isfinite(part_rho) and
              part_rho >= .5)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(per.clean_drop, per.shared_drop, s=70, color="#4c78a8")
    for row in per.itertuples():
        ax.annotate(row.part, (row.clean_drop, row.shared_drop), xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(0, color="black", lw=1); ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("clean renderer deletion: probability drop")
    ax.set_ylabel("shared mask deletion: probability drop")
    ax.set_title("FunnyBird calibration of the shared deletion")
    plt.tight_layout(); fig.savefig(out / "funnybird_deletion_calibration.png", dpi=180)
    plt.close(fig)
    return {"status": "PASS" if passed else "FAIL", "matched_rows": len(merged),
            "row_spearman": row_rho, "part_spearman": part_rho,
            "clean_positive_parts": clean_positive,
            "shared_positive_parts": shared_positive,
            "rule": "n>=100; >=4/5 parts positive in both; row rho>0; part rho>=0.5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds", required=True)
    ap.add_argument("--cub70", required=True)
    ap.add_argument("--clean-funnybirds", required=True,
                    help="accepted renderer-deletion FunnyBird parquet")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fb, cub = load(args.funnybirds, "funnybirds"), load(args.cub70, "cub70")
    calibration = calibrate_funnybird(fb, args.clean_funnybirds, out)
    both = pd.concat([fb, cub], ignore_index=True)
    scales = (both.groupby(["dataset", "concept_name"]).z_original.std()
              .rename("original_z_sd").reset_index())
    both = both.merge(scales, on=["dataset", "concept_name"], how="left")
    both["adjusted_drop_std"] = (
        both.target_minus_control_z /
        both.original_z_sd.where(both.original_z_sd >= .1, .1)
    )
    both["target_hurts_more"] = both.target_minus_control_z < 0
    both["context_still_positive"] = both.z_target_deleted > 0
    both["context_beats_part_only"] = both.context_minus_part_only_z > 0

    summary = (both.groupby(["dataset", "part_common"]).agg(
        n=("image", "size"), n_images=("image", "nunique"),
        n_concepts=("concept_name", "nunique"), n_species=("class_label", "nunique"),
        median_raw_adjusted_drop=("target_minus_control_z", "median"),
        median_standardized_drop=("adjusted_drop_std", "median"),
        target_hurts_more_rate=("target_hurts_more", "mean"),
        context_still_positive_rate=("context_still_positive", "mean"),
        context_beats_part_only_rate=("context_beats_part_only", "mean"),
    ).reset_index())
    summary.to_csv(out / "paired_deletion_summary.csv", index=False)

    species_rows = []
    grouped = both.groupby(["dataset", "part_common", "concept_name"])
    for group_number, ((dataset_name, part, concept), group) in enumerate(grouped):
        values = group.z_target_deleted.to_numpy()
        labels = group.class_label.to_numpy()
        value = species_omega2(values, labels)
        if np.isfinite(value):
            rng = np.random.default_rng(20260801 + group_number)
            null = np.asarray([species_omega2(values, rng.permutation(labels))
                               for _ in range(200)])
            null = null[np.isfinite(null)]
            p = float((1 + (null >= value).sum()) / (1 + len(null)))
            species_rows.append({"dataset": dataset_name, "part_common": part,
                                 "concept_name": concept, "species_omega2": value,
                                 "permutation_p": p,
                                 "n_species": group.class_label.nunique(),
                                 "n": len(group)})
    species = pd.DataFrame(species_rows)
    species.to_csv(out / "paired_deletion_species_residual.csv", index=False)

    palette = {"funnybirds": "#7b3294", "cub70": "#008837"}
    order = [p for p in PART_ORDER if p in set(both.part_common)]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    sns.boxplot(data=both, x="part_common", y="adjusted_drop_std", hue="dataset",
                order=order, hue_order=["funnybirds", "cub70"], palette=palette,
                showfliers=False, ax=axes[0])
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("mapped part"); axes[0].set_ylabel("target minus control deletion (raw z / concept SD)")
    axes[0].set_title("Same test: does deleting the named part hurt more?")
    rate_specs = [
        ("target_hurts_more_rate", "Target deletion hurts more"),
        ("context_still_positive_rate", "Score stays positive after deletion"),
        ("context_beats_part_only_rate", "Deleted-part context beats part-preserved input"),
    ]
    for ax, (column, title) in zip(axes[1:], rate_specs):
        sns.barplot(data=summary, x="part_common", y=column, hue="dataset",
                    order=order, hue_order=["funnybirds", "cub70"], palette=palette,
                    errorbar=None, ax=ax)
        ax.set_ylim(0, 1); ax.set_xlabel("mapped part"); ax.set_ylabel("fraction")
        ax.set_title(title)
    plt.tight_layout(); fig.savefig(out / "paired_deletion_main.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(species):
        sns.boxplot(data=species, x="part_common", y="species_omega2", hue="dataset",
                    order=order, hue_order=["funnybirds", "cub70"], palette=palette,
                    showfliers=False, ax=ax)
    ax.set_ylim(0, 1); ax.set_xlabel("mapped part")
    ax.set_ylabel("bias-corrected source-species effect on deleted-part score")
    ax.set_title("After fixing the exact concept, does source species still matter?")
    plt.tight_layout(); fig.savefig(out / "paired_deletion_species.png", dpi=180); plt.close(fig)

    audit = {
        "status": "PASS", "funnybirds_rows": len(fb), "cub70_rows": len(cub),
        "shared_parts": order,
        "funnybird_calibration": calibration,
        "warning": "Species omega-squared and permutation p are observational; context ablation and deletion controls carry the causal burden.",
    }
    (out / "paired_deletion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(f"[PAIRED DELETION COMPARISON PASS] -> {out}")


if __name__ == "__main__":
    main()
