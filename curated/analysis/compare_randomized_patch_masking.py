#!/usr/bin/env python3
"""Calibrate and summarize the randomized small-mask dose-response experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


KEY = ["dataset", "image_index", "class_label", "part", "concept_idx",
       "concept_name", "repeat", "fill", "requested_dose"]
REQUIRED = set(KEY) | {
    "location", "z_original", "z_masked", "delta_z", "p_original",
    "p_masked", "drop_p", "score_still_positive", "mask_alpha_mass",
    "target_coverage", "damage_per_target_area", "rgb_mae",
    "rgb_changed_fraction",
}
PART_ORDER = ["tail", "wing", "beak", "foot", "leg", "eye", "head",
              "body", "neck"]
PALETTE = {"target": "#c23b3b", "other_bird": "#2878b5",
           "background": "#2d9348"}


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


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.pivot(index=KEY, columns="location",
                         values=["delta_z", "drop_p", "p_original", "p_masked",
                                 "score_still_positive",
                                 "damage_per_target_area", "rgb_mae",
                                 "rgb_changed_fraction"]).reset_index()
    values.columns = ["_".join(str(v) for v in col if str(v))
                      if isinstance(col, tuple) else col for col in values.columns]
    needed = [f"{metric}_{location}" for metric in
              ["delta_z", "drop_p", "p_masked", "damage_per_target_area"]
              for location in ["target", "other_bird", "background"]]
    if values[needed].isna().any().any():
        raise ValueError("target/control pairing is incomplete")
    values["adjusted_delta_z"] = (
        values.delta_z_target - values.delta_z_other_bird)
    values["adjusted_drop_p"] = (
        values.drop_p_target - values.drop_p_other_bird)
    values["background_adjusted_drop_p"] = (
        values.drop_p_target - values.drop_p_background)
    values["target_retained_p"] = (
        values.p_masked_target / values.p_original_target.clip(.001))
    values["part_common"] = values.part.replace({"foot": "leg"})
    return values


def slope_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "image_index", "class_label", "part",
                  "concept_idx", "concept_name", "repeat", "fill", "location"]
    for key, group in frame.groupby(group_cols):
        group = group.sort_values("requested_dose")
        if group.requested_dose.nunique() < 3:
            continue
        x = group.requested_dose.to_numpy(float)
        rows.append(dict(zip(group_cols, key)) | {
            "drop_p_slope": float(np.polyfit(x, group.drop_p, 1)[0]),
            "delta_z_slope": float(np.polyfit(x, group.delta_z, 1)[0]),
        })
    slopes = pd.DataFrame(rows)
    if slopes.empty:
        raise ValueError("no dose slopes could be estimated")
    return slopes


def calibration(fb: pd.DataFrame, clean_path: str, out: Path) -> dict:
    clean = pd.read_parquet(clean_path).copy()
    required = {"image_idx", "part", "typ_concept", "p_intact", "p_removed",
                "changed_frac"}
    missing = required - set(clean)
    if missing:
        raise ValueError(f"clean reference lacks {sorted(missing)}")
    clean = clean[clean.changed_frac > 1e-3].copy()
    clean["clean_drop"] = clean.p_intact - clean.p_removed

    paired_fb = paired(fb)
    max_dose = paired_fb.requested_dose.max()
    high = paired_fb[paired_fb.requested_dose == max_dose].copy()
    matched = high.merge(
        clean[["image_idx", "part", "typ_concept", "clean_drop"]],
        left_on=["image_index", "part", "concept_idx"],
        right_on=["image_idx", "part", "typ_concept"], how="inner",
        validate="many_to_one",
    )
    slopes = slope_rows(fb)
    slope_pivot = slopes.pivot_table(
        index=["dataset", "image_index", "class_label", "part", "concept_idx",
               "concept_name", "repeat", "fill"],
        columns="location", values="drop_p_slope").reset_index()
    slope_pivot["adjusted_slope"] = (
        slope_pivot.target - slope_pivot.other_bird)

    part_rows, fill_rows = [], []
    for fill in sorted(fb.fill.unique()):
        local = matched[matched.fill == fill]
        per_part = (local.groupby("part").agg(
            clean_drop=("clean_drop", "median"),
            target_drop=("drop_p_target", "median"),
            control_drop=("drop_p_other_bird", "median"),
            adjusted_drop=("adjusted_drop_p", "median"),
            n=("image_index", "size"),
        ).reset_index())
        local_slopes = (slope_pivot[slope_pivot.fill == fill]
                        .groupby("part").adjusted_slope.median())
        per_part["adjusted_slope"] = per_part.part.map(local_slopes)
        per_part["fill"] = fill
        part_rows.append(per_part)
        row_rho = float(local[["clean_drop", "drop_p_target"]]
                        .corr(method="spearman").iloc[0, 1])
        fill_rows.append({"fill": fill, "matched_rows": len(local),
                          "row_spearman_clean_vs_patch": row_rho})
    parts = pd.concat(part_rows, ignore_index=True)
    parts.to_csv(out / "funnybird_patch_calibration_by_part.csv", index=False)

    fill_pivot = parts.pivot(index="part", columns="fill", values="adjusted_drop")
    if set(["local_blur", "local_mean"]).issubset(fill_pivot):
        fill_rho = float(fill_pivot[["local_blur", "local_mean"]]
                         .corr(method="spearman").iloc[0, 1])
    else:
        fill_rho = float("nan")

    required_parts = ["wing", "foot"]
    observed_parts = set(parts.part)
    checks = {
        "all_five_funnybird_parts_present":
            {"tail", "wing", "beak", "foot", "eye"} <= observed_parts,
        "wing_and_foot_target_hurt_more_both_fills": bool(
            all((parts[(parts.fill == fill) & parts.part.isin(required_parts)]
                 .set_index("part").adjusted_drop > 0).reindex(required_parts).fillna(False).all()
                for fill in ["local_blur", "local_mean"])),
        "wing_and_foot_positive_adjusted_dose_slope_both_fills": bool(
            all((parts[(parts.fill == fill) & parts.part.isin(required_parts)]
                 .set_index("part").adjusted_slope > 0).reindex(required_parts).fillna(False).all()
                for fill in ["local_blur", "local_mean"])),
        "fill_part_order_agreement_at_least_0p5": bool(
            np.isfinite(fill_rho) and fill_rho >= .5),
        "clean_row_direction_positive_both_fills": bool(
            all(np.isfinite(row["row_spearman_clean_vs_patch"]) and
                row["row_spearman_clean_vs_patch"] > 0 for row in fill_rows)),
        "at_least_four_parts_target_drop_positive_each_fill": bool(
            all(int((parts[parts.fill == fill].target_drop > 0).sum()) >= 4
                for fill in ["local_blur", "local_mean"])),
    }
    passed = all(checks.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.scatterplot(data=parts, x="clean_drop", y="target_drop", hue="part",
                    style="fill", s=90, ax=axes[0])
    axes[0].axhline(0, color="black", lw=1); axes[0].axvline(0, color="black", lw=1)
    axes[0].set_title("FunnyBird calibration: clean deletion vs small masks")
    axes[0].set_xlabel("clean renderer deletion: probability drop")
    axes[0].set_ylabel("highest-dose target patches: probability drop")
    sns.scatterplot(data=parts, x="adjusted_drop", y="adjusted_slope", hue="part",
                    style="fill", s=90, ax=axes[1], legend=False)
    axes[1].axhline(0, color="black", lw=1); axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title("Must hurt more than other-bird patches and grow with dose")
    axes[1].set_xlabel("target drop minus other-bird drop")
    axes[1].set_ylabel("dose slope of that adjusted drop")
    plt.tight_layout(); fig.savefig(out / "funnybird_patch_calibration.png", dpi=180)
    plt.close(fig)
    return {
        "status": "PASS" if passed else "FAIL", "max_dose": float(max_dose),
        "matched_rows": int(len(matched)), "fill_part_order_spearman": fill_rho,
        "fills": fill_rows, "checks": checks,
        "rule": "all checks must pass before CUB70 runs or is interpreted",
    }


def dataset_plots(frame: pd.DataFrame, out: Path) -> dict:
    dataset_name = str(frame.dataset.iloc[0])
    order = [part for part in PART_ORDER if part in set(frame.part_common)]
    # Raw-z slopes are divided by the ordinary within-concept z spread only for
    # cross-concept display. Raw components remain in the parquet.
    scales = (frame.groupby("concept_name").z_original.std()
              .clip(lower=.1).rename("z_sd"))
    frame = frame.join(scales, on="concept_name")
    frame["delta_z_std"] = frame.delta_z / frame.z_sd

    fig, axes = plt.subplots(len(order), 2, figsize=(12, max(4, 2.7 * len(order))),
                             squeeze=False)
    for row, part in enumerate(order):
        local = frame[frame.part_common == part]
        for col, fill in enumerate(["local_blur", "local_mean"]):
            ax = axes[row, col]
            sub = local[local.fill == fill]
            sns.lineplot(data=sub, x="requested_dose", y="delta_z_std",
                         hue="location", hue_order=["target", "other_bird", "background"],
                         palette=PALETTE, estimator="median", errorbar=None, marker="o", ax=ax)
            ax.axhline(0, color="black", lw=.8)
            ax.set_title(f"{part} · {fill}")
            ax.set_xlabel("requested fraction of target-part area")
            ax.set_ylabel("masked − original raw z / ordinary z SD")
            if row or col:
                legend = ax.get_legend()
                if legend: legend.remove()
    plt.tight_layout(); fig.savefig(out / f"{dataset_name}_patch_dose_raw_z.png", dpi=180)
    plt.close(fig)

    pair = paired(frame)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    sns.lineplot(data=pair, x="requested_dose", y="adjusted_drop_p", hue="part_common",
                 hue_order=order, style="fill", estimator="median", errorbar=None,
                 marker="o", ax=axes[0])
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_title(f"{dataset_name}: does target masking hurt more?")
    axes[0].set_xlabel("requested fraction of target-part area")
    axes[0].set_ylabel("probability drop: target minus other-bird control")
    sns.lineplot(data=pair, x="requested_dose", y="target_retained_p", hue="part_common",
                 hue_order=order, style="fill", estimator="median", errorbar=None,
                 marker="o", ax=axes[1])
    axes[1].axhline(1, color="black", lw=1)
    axes[1].set_title("How much concept probability remains?")
    axes[1].set_xlabel("requested fraction of target-part area")
    axes[1].set_ylabel("target-masked probability / original probability")
    plt.tight_layout(); fig.savefig(out / f"{dataset_name}_patch_dose_summary.png", dpi=180)
    plt.close(fig)

    high = pair[pair.requested_dose == pair.requested_dose.max()]
    summary = (high.groupby(["part_common", "fill"]).agg(
        n=("image_index", "size"), n_images=("image_index", "nunique"),
        n_concepts=("concept_name", "nunique"),
        median_adjusted_drop_p=("adjusted_drop_p", "median"),
        median_retained_p=("target_retained_p", "median"),
        median_target_rgb_mae=("rgb_mae_target", "median"),
        median_other_bird_rgb_mae=("rgb_mae_other_bird", "median"),
        target_hurts_more_rate=("adjusted_drop_p", lambda x: float((x > 0).mean())),
        score_still_positive_rate=("score_still_positive_target", "mean"),
    ).reset_index())
    summary.to_csv(out / f"{dataset_name}_patch_summary.csv", index=False)
    return {"rows": len(frame), "images": int(frame.image_index.nunique()),
            "parts": order, "max_dose": float(pair.requested_dose.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds", required=True)
    ap.add_argument("--clean-funnybirds", required=True)
    ap.add_argument("--cub70")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fail-on-calibration", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fb = load(args.funnybirds, "funnybirds")
    cal = calibration(fb, args.clean_funnybirds, out)
    audit = {"status": cal["status"], "funnybird_calibration": cal,
             "funnybirds": dataset_plots(fb, out),
             "claim_limit": "robust local pixel reliance and partial-context retention; not a causal swap"}
    if args.cub70:
        if cal["status"] != "PASS":
            raise RuntimeError("FunnyBird calibration failed; refusing CUB70 analysis")
        cub = load(args.cub70, "cub70")
        audit["cub70"] = dataset_plots(cub, out)
        audit["status"] = "PASS"
    (out / "randomized_patch_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[FUNNYBIRD PATCH CALIBRATION {cal['status']}] -> {out}")
    if args.cub70:
        print(f"[CROSS-DATASET PATCH ANALYSIS PASS] -> {out}")
    if args.fail_on_calibration and cal["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
