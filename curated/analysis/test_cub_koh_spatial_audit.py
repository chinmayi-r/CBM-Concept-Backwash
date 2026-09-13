#!/usr/bin/env python3
"""Behavioral tests for the CUB Koh spatial audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cub_koh_spatial_audit import (
    cross_fitted_label_means,
    localization_metrics,
    saved_head_use_table,
)


def test_localization_metrics() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    inside = np.zeros((4, 4)); inside[:2, :2] = 1
    outside = np.zeros((4, 4)); outside[2:, 2:] = 1
    good = localization_metrics(inside, mask)
    bad = localization_metrics(outside, mask)
    assert np.isclose(good["attribution_mass_inside_mask"], 1.0)
    assert np.isclose(good["area_adjusted_enrichment"], 4.0)
    assert good["pointing_inside_mask"] == 1
    assert np.isclose(good["equal_area_iou"], 1.0)
    assert bad["attribution_mass_inside_mask"] == 0
    assert bad["pointing_inside_mask"] == 0
    empty = localization_metrics(np.zeros((4, 4)), mask)
    assert np.isnan(empty["pointing_inside_mask"])
    assert empty["has_spatial_signal"] == 0


def test_cross_fitted_means_do_not_use_held_out_rows() -> None:
    z = np.array([[0.0], [2.0], [10.0], [14.0]])
    c = np.array([[0], [0], [0], [0]])
    folds = np.array([0, 0, 1, 1])
    replaced = cross_fitted_label_means(z, c, folds)
    assert np.allclose(replaced[:2, 0], 12.0)
    assert np.allclose(replaced[2:, 0], 1.0)


def test_saved_head_use_detects_used_magnitude() -> None:
    rows = []
    z_rows = [[-3.0, -1.0], [-2.0, -1.0], [2.0, 1.0], [3.0, 1.0],
              [-4.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [4.0, 1.0],
              [-5.0, -1.0], [-0.5, -1.0]]
    weight = np.array([[-1.0, 0.0], [1.0, 0.0]])
    bias = np.zeros(2)
    for image, values in enumerate(z_rows):
        logits = np.asarray(values) @ weight.T
        pred = int(np.argmax(logits))
        for concept, value in enumerate(values):
            rows.append({
                "image": f"image_{image}", "concept_index": concept,
                "concept_name": f"has_tail_shape::value_{concept}",
                "z": value, "gt_label": int(value > 0),
                "y_true": pred, "y_pred": pred,
            })
    table = saved_head_use_table(pd.DataFrame(rows), weight, bias)
    assert set(table.block) >= {"all 112", "tail"}
    assert table.mean_absolute_class_logit_change.max() > 0


if __name__ == "__main__":
    test_localization_metrics()
    test_cross_fitted_means_do_not_use_held_out_rows()
    test_saved_head_use_detects_used_magnitude()
    print("CUB KOH SPATIAL AUDIT SYNTHETIC PASS")
