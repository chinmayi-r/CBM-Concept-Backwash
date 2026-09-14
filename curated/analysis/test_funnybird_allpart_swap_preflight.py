#!/usr/bin/env python3
"""Behavioral tests for the all-part controlled-swap preflight."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from funnybird_allpart_swap_preflight_core import (  # noqa: E402
    balanced_swap_rows, best_other_margin, decompose_swap, matched_swap_rows,
    summarize_pathways,
)


def test_balanced_selection() -> None:
    rows = []
    for part, width in (("tail", 3), ("wing", 2)):
        for donor in range(width):
            for image in range(4):
                rows.append({
                    "part": part, "var_src": (donor + 1) % width,
                    "var_donor": donor, "orig_render_id": f"{part}-o{image}",
                    "render_id": f"{part}-d{donor}-i{image}",
                    "image_cf_sha256": f"{part}-{donor}-{image}",
                })
    selected = balanced_swap_rows(pd.DataFrame(rows), ["tail", "wing"], 6)
    assert selected.groupby("part").size().to_dict() == {"tail": 6, "wing": 6}
    assert selected.groupby("part").var_donor.nunique().to_dict() == {
        "tail": 3, "wing": 2}


def test_decomposition_and_signs() -> None:
    spans = {"tail": (0, 3), "wing": (3, 5)}
    z_orig = np.array([4.0, -2.0, -3.0, 3.0, -1.0])
    z_cf = np.array([1.0, 2.0, -2.0, 2.0, 1.0])
    # Source-minus-donor weights: [1, 0, 0, 2, -1].
    weights = np.array([[1.0, 0.0, 0.0, 2.0, -1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
    rows = decompose_swap(
        z_orig=z_orig, z_cf=z_cf, class_weights=weights,
        source_species=0, donor_species=1, input_part="tail",
        source_value=0, donor_value=1,
        original_values={"tail": 0, "wing": 0}, spans=spans)
    tail, wing = rows
    # Tail donor-vs-source margin: -6 before, +1 after, response +7.
    assert tail["target_response_delta"] == 7.0
    # Wing true-vs-best margin: 4 before, 1 after, damage -3.
    assert wing["unchanged_margin_change"] == -3.0
    # Tail contributes -3 to source-minus-donor gap; wing contributes -4.
    assert tail["class_gap_shift"] == -3.0
    assert wing["class_gap_shift"] == -4.0
    assert tail["total_class_gap_shift"] == -7.0
    assert best_other_margin(np.array([3.0, -1.0]), 0) == 4.0


def test_matched_selection() -> None:
    candidate = pd.DataFrame([
        dict(render_id="r2", part="wing", var_src=0, var_donor=1,
             sid_src=2, sid_donor=3, orig_render_id="o2",
             image_orig_sha256="a2", image_cf_sha256="b2", value=20),
        dict(render_id="r1", part="tail", var_src=1, var_donor=2,
             sid_src=0, sid_donor=1, orig_render_id="o1",
             image_orig_sha256="a1", image_cf_sha256="b1", value=10),
    ])
    selection = candidate.iloc[[1, 0]].drop(columns="value")
    matched = matched_swap_rows(candidate, selection, ["tail", "wing"])
    assert matched.render_id.tolist() == ["r1", "r2"]
    assert matched.value.tolist() == [10, 20]
    broken = selection.copy()
    broken.loc[0, "image_cf_sha256"] = "wrong"
    try:
        matched_swap_rows(candidate, broken, ["tail", "wing"])
    except ValueError as exc:
        assert "image_cf_sha256" in str(exc)
    else:
        raise AssertionError("mismatched image hash was accepted")


def test_summary() -> None:
    frame = pd.DataFrame([
        dict(input_part="tail", output_part="tail", original_image="o1",
             render_id="r1", mean_absolute_score_change=2.0,
             binary_flip_fraction=0.5, class_gap_shift=-1.0,
             target_response_delta=4.0, unchanged_margin_change=np.nan),
        dict(input_part="tail", output_part="tail", original_image="o2",
             render_id="r2", mean_absolute_score_change=4.0,
             binary_flip_fraction=0.0, class_gap_shift=1.0,
             target_response_delta=2.0, unchanged_margin_change=np.nan),
    ])
    summary = summarize_pathways(frame).iloc[0]
    assert summary.n_rows == 2
    assert summary.mean_absolute_score_change == 3.0
    assert summary.mean_target_response_delta == 3.0
    assert summary.fraction_class_gap_shift_sourceward == 0.5


if __name__ == "__main__":
    test_balanced_selection()
    test_decomposition_and_signs()
    test_matched_selection()
    test_summary()
    print("FUNNYBIRD ALL-PART SWAP PREFLIGHT SYNTHETIC PASS")
