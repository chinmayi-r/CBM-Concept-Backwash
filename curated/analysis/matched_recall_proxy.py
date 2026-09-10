#!/usr/bin/env python3
"""Matched species diagnostics and FunnyBird calibration for Notebook 05.

The functions here are deliberately independent of notebooks.  They operate on
already-exported raw concept logits and never train or alter a scientific model.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


PAIR_COLUMNS = [
    "concept_name",
    "species_a",
    "species_b",
    "matched_positive_n",
    "matched_negative_n",
    "recall_gap",
    "balanced_accuracy_gap",
    "positive_raw_z_gap",
    "label_conditioned_raw_z_gap",
]


def _required(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"matched-recall input is missing columns: {sorted(missing)}")


def matched_species_diagnostics(
    frame: pd.DataFrame,
    *,
    concept_col: str = "concept_name",
    species_col: str = "y_true",
    label_col: str = "gt_label",
    score_col: str = "z",
    min_each: int = 3,
    max_pairs_per_concept: int = 50,
    bootstrap_repeats: int = 200,
    seed: int = 20260910,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure species differences after positive/negative support is matched.

    A species is eligible for an exact concept only when it has at least
    ``min_each`` positive and negative images.  For each species pair we match
    positive and negative counts, bootstrap both classes, and calculate:

    * positive recall gap: ``abs(TPR_A - TPR_B)``;
    * balanced-accuracy gap: ``abs(BA_A - BA_B)``;
    * positive raw-z gap: ``abs(mean(z_pos,A) - mean(z_pos,B))``;
    * label-conditioned raw-z gap: the mean of the positive and negative gaps.

    No diagnostic classifier is fitted.  The prediction is the accepted CBM's
    own threshold rule ``z > 0``.
    """
    if min_each < 1 or max_pairs_per_concept < 1 or bootstrap_repeats < 1:
        raise ValueError("min_each, max_pairs_per_concept, and bootstrap_repeats must be positive")
    _required(frame, {concept_col, species_col, label_col, score_col})
    local = frame[[concept_col, species_col, label_col, score_col]].copy()
    local.columns = ["concept_name", "species", "label", "z"]
    local["label"] = local.label.astype(int)
    if not set(local.label.unique()).issubset({0, 1}):
        raise ValueError("matched-recall labels must be binary")
    if not np.isfinite(local.z.to_numpy(dtype=float)).all():
        raise ValueError("matched-recall scores contain non-finite values")

    rng = np.random.default_rng(seed)
    pair_rows: list[dict] = []
    eligibility_rows: list[dict] = []
    for concept_name, concept in local.groupby("concept_name", sort=True):
        eligible: list[tuple[object, np.ndarray, np.ndarray]] = []
        for species, group in concept.groupby("species", sort=True):
            positive = group.loc[group.label == 1, "z"].to_numpy(dtype=float)
            negative = group.loc[group.label == 0, "z"].to_numpy(dtype=float)
            if len(positive) >= min_each and len(negative) >= min_each:
                eligible.append((species, positive, negative))
        candidates = list(combinations(range(len(eligible)), 2))
        if len(candidates) > max_pairs_per_concept:
            selected = np.sort(rng.choice(len(candidates), max_pairs_per_concept, replace=False))
            candidates = [candidates[int(index)] for index in selected]
        eligibility_rows.append(
            {
                "concept_name": concept_name,
                "eligible_species": len(eligible),
                "candidate_pairs": len(list(combinations(range(len(eligible)), 2))),
                "used_pairs": len(candidates),
            }
        )
        for left, right in candidates:
            species_a, positive_a, negative_a = eligible[left]
            species_b, positive_b, negative_b = eligible[right]
            n_positive = min(len(positive_a), len(positive_b))
            n_negative = min(len(negative_a), len(negative_b))
            positive_a_boot = positive_a[
                rng.integers(len(positive_a), size=(bootstrap_repeats, n_positive))
            ]
            positive_b_boot = positive_b[
                rng.integers(len(positive_b), size=(bootstrap_repeats, n_positive))
            ]
            negative_a_boot = negative_a[
                rng.integers(len(negative_a), size=(bootstrap_repeats, n_negative))
            ]
            negative_b_boot = negative_b[
                rng.integers(len(negative_b), size=(bootstrap_repeats, n_negative))
            ]

            tpr_a = (positive_a_boot > 0).mean(axis=1)
            tpr_b = (positive_b_boot > 0).mean(axis=1)
            tnr_a = (negative_a_boot <= 0).mean(axis=1)
            tnr_b = (negative_b_boot <= 0).mean(axis=1)
            ba_a = 0.5 * (tpr_a + tnr_a)
            ba_b = 0.5 * (tpr_b + tnr_b)
            positive_z_gap = np.abs(positive_a_boot.mean(axis=1) - positive_b_boot.mean(axis=1))
            negative_z_gap = np.abs(negative_a_boot.mean(axis=1) - negative_b_boot.mean(axis=1))
            pair_rows.append(
                {
                    "concept_name": concept_name,
                    "species_a": species_a,
                    "species_b": species_b,
                    "matched_positive_n": n_positive,
                    "matched_negative_n": n_negative,
                    "recall_gap": float(np.abs(tpr_a - tpr_b).mean()),
                    "balanced_accuracy_gap": float(np.abs(ba_a - ba_b).mean()),
                    "positive_raw_z_gap": float(positive_z_gap.mean()),
                    "label_conditioned_raw_z_gap": float(
                        (0.5 * (positive_z_gap + negative_z_gap)).mean()
                    ),
                }
            )

    pairs = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
    eligibility = pd.DataFrame(eligibility_rows)
    if pairs.empty:
        summary = pd.DataFrame(
            columns=[
                "concept_name",
                "n_species_pairs",
                "mean_recall_gap",
                "mean_balanced_accuracy_gap",
                "mean_positive_raw_z_gap",
                "mean_label_conditioned_raw_z_gap",
                "min_matched_positive_n",
                "min_matched_negative_n",
            ]
        )
    else:
        summary = (
            pairs.groupby("concept_name", as_index=False)
            .agg(
                n_species_pairs=("recall_gap", "size"),
                mean_recall_gap=("recall_gap", "mean"),
                mean_balanced_accuracy_gap=("balanced_accuracy_gap", "mean"),
                mean_positive_raw_z_gap=("positive_raw_z_gap", "mean"),
                mean_label_conditioned_raw_z_gap=("label_conditioned_raw_z_gap", "mean"),
                min_matched_positive_n=("matched_positive_n", "min"),
                min_matched_negative_n=("matched_negative_n", "min"),
            )
            .sort_values("concept_name")
            .reset_index(drop=True)
        )
    return pairs, summary, eligibility


def funnybird_swap_targets(swaps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate controlled FunnyBird outcomes by inserted exact concept."""
    _required(
        swaps,
        {"part", "var_donor", "response_delta", "margin"},
    )
    local = swaps.copy()
    local["concept_name"] = local.part.astype(str) + "_" + local.var_donor.astype(int).astype(str)
    local["controlled_event"] = (local.response_delta > 0) & (local.margin < 0)
    local["donor_win"] = local.margin > 0
    return (
        local.groupby(["part", "concept_name"], as_index=False)
        .agg(
            swap_rows=("margin", "size"),
            controlled_event_rate=("controlled_event", "mean"),
            donor_win_rate=("donor_win", "mean"),
            median_response_delta=("response_delta", "median"),
            median_final_margin=("margin", "median"),
        )
        .assign(donor_failure_rate=lambda d: 1.0 - d.donor_win_rate)
    )


def calibrate_recall_warning(
    recall_summary: pd.DataFrame,
    swap_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compare ordinary-image recall gaps with known controlled swap failures.

    The predeclared primary association is Spearman correlation between mean
    matched positive-recall gap and controlled-event rate.  A provisional
    warning proxy additionally requires a positive within-part-centred
    association and positive leave-one-part-out correlations for at least four
    of the five omitted-part analyses.  Otherwise the method is not calibrated.
    """
    _required(recall_summary, {"concept_name", "mean_recall_gap"})
    _required(swap_targets, {"concept_name", "part", "controlled_event_rate"})
    merged = recall_summary.merge(swap_targets, on="concept_name", how="inner", validate="one_to_one")
    if len(merged) < 5:
        raise ValueError(f"recall calibration has only {len(merged)} aligned exact concepts")
    merged = merged.copy()
    merged["recall_gap_within_part"] = merged.mean_recall_gap - merged.groupby("part").mean_recall_gap.transform("mean")
    merged["event_rate_within_part"] = merged.controlled_event_rate - merged.groupby("part").controlled_event_rate.transform("mean")
    rows = [
        {
            "check": "all exact concepts",
            "omitted_part": "none",
            "n_concepts": len(merged),
            "spearman_recall_vs_controlled_event": merged.mean_recall_gap.corr(
                merged.controlled_event_rate, method="spearman"
            ),
        },
        {
            "check": "within-part centred",
            "omitted_part": "none",
            "n_concepts": len(merged),
            "spearman_recall_vs_controlled_event": merged.recall_gap_within_part.corr(
                merged.event_rate_within_part, method="spearman"
            ),
        },
    ]
    expected_parts = ["tail", "wing", "beak", "foot", "eye"]
    observed_parts = set(merged.part.unique())
    for part in expected_parts:
        subset = merged[merged.part != part]
        rows.append(
            {
                "check": "leave one part out",
                "omitted_part": part,
                "n_concepts": len(subset),
                "spearman_recall_vs_controlled_event": subset.mean_recall_gap.corr(
                    subset.controlled_event_rate, method="spearman"
                ),
            }
        )
    checks = pd.DataFrame(rows)
    overall = float(checks.loc[checks.check == "all exact concepts", "spearman_recall_vs_controlled_event"].iloc[0])
    centred = float(checks.loc[checks.check == "within-part centred", "spearman_recall_vs_controlled_event"].iloc[0])
    leave_one_out = checks.loc[checks.check == "leave one part out", "spearman_recall_vs_controlled_event"]
    positive_leave_one_out = int((leave_one_out > 0).sum())
    if observed_parts == set(expected_parts) and overall > 0 and centred > 0 and positive_leave_one_out >= 4:
        verdict = "PROVISIONAL ORDINAL WARNING PROXY"
    else:
        verdict = "METHOD NOT CALIBRATED AS A BACKWASH PROXY"
    return merged, checks, verdict
