#!/usr/bin/env python3
"""Synthetic checks for the Notebook 05 matched-recall calibration helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from matched_recall_proxy import (
    calibrate_recall_warning,
    funnybird_swap_targets,
    matched_species_eligibility,
    matched_species_diagnostics,
)


def fixture() -> pd.DataFrame:
    rows = []
    for concept_index in range(6):
        concept = f"tail_{concept_index}"
        for species in range(4):
            for label in (0, 1):
                for image in range(4):
                    # Species 0/1 are deliberately easier than 2/3, leaving a
                    # measurable recall and balanced-accuracy gap.
                    direction = 1 if label else -1
                    score = direction * (2.0 if species < 2 else -0.2) + 0.03 * image
                    rows.append(
                        {
                            "concept_name": concept,
                            "y_true": species,
                            "gt_label": label,
                            "z": score,
                        }
                    )
    return pd.DataFrame(rows)


def test_matching() -> None:
    audit = matched_species_eligibility(fixture())
    assert audit.candidate_species_pairs.tolist() == [36, 36, 36]
    pairs, summary, eligibility = matched_species_diagnostics(
        fixture(), bootstrap_repeats=20, seed=7
    )
    assert len(summary) == 6
    assert (eligibility.eligible_species == 4).all()
    assert (pairs.matched_positive_n == 4).all()
    assert (pairs.matched_negative_n == 4).all()
    assert (summary.mean_recall_gap > 0).all()
    assert (summary.mean_balanced_accuracy_gap > 0).all()
    assert (summary.mean_label_conditioned_raw_z_gap > 0).all()


def test_calibration_contract() -> None:
    _, summary, _ = matched_species_diagnostics(fixture(), bootstrap_repeats=20, seed=7)
    summary = summary.copy()
    summary["mean_recall_gap"] = np.linspace(0.05, 0.8, len(summary))
    swaps = []
    for index in range(6):
        for row in range(20):
            event = row < (index + 1) * 2
            swaps.append(
                {
                    "part": "tail",
                    "var_donor": index,
                    "response_delta": 1.0 if event else -1.0,
                    "margin": -1.0 if event else 1.0,
                }
            )
    targets = funnybird_swap_targets(pd.DataFrame(swaps))
    merged, checks, verdict = calibrate_recall_warning(summary, targets)
    assert len(merged) == 6
    assert checks.iloc[0].spearman_recall_vs_controlled_event > 0
    # One-part synthetic data cannot satisfy the five-part robustness rule.
    assert verdict == "METHOD NOT CALIBRATED AS A BACKWASH PROXY"


def test_five_part_calibration_can_pass() -> None:
    parts = ["tail", "wing", "beak", "foot", "eye"]
    summary_rows = []
    swap_rows = []
    for part_index, part in enumerate(parts):
        for value in range(3):
            concept_name = f"{part}_{value}"
            signal = 0.05 + 0.08 * value + 0.01 * part_index
            summary_rows.append(
                {"concept_name": concept_name, "mean_recall_gap": signal}
            )
            event_rows = int(round(signal * 100))
            for row in range(100):
                event = row < event_rows
                swap_rows.append(
                    {
                        "part": part,
                        "var_donor": value,
                        "response_delta": 1.0 if event else -1.0,
                        "margin": -1.0 if event else 1.0,
                    }
                )
    _, checks, verdict = calibrate_recall_warning(
        pd.DataFrame(summary_rows), funnybird_swap_targets(pd.DataFrame(swap_rows))
    )
    assert (checks.spearman_recall_vs_controlled_event > 0).all()
    assert verdict == "PROVISIONAL ORDINAL WARNING PROXY"


if __name__ == "__main__":
    test_matching()
    test_calibration_contract()
    test_five_part_calibration_can_pass()
    print("MATCHED RECALL PROXY SYNTHETIC PASS")
