#!/usr/bin/env python3
"""Small regression tests for the focused follow-up calculations.

These tests target earlier failure modes.  They are not a substitute for the
strict real-artifact checks performed by funnybird_followup_diagnostics.py.
"""
from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import funnybird_followup_diagnostics as followup


class FollowupRegressionTests(unittest.TestCase):
    def test_controlled_event_uses_strict_final_negative_margin(self):
        frame = pd.DataFrame(
            {
                "response_delta": [1.0, 1.0, 0.0, -1.0],
                "m_cf": [-0.1, 0.0, -0.1, -0.1],
            }
        )
        self.assertEqual(
            followup.controlled_event(frame).tolist(), [True, False, False, False]
        )

    def test_record_pair_may_change_only_attribute_label(self):
        left = {"img_path": "a.png", "class_label": 3, "attribute_label": [1, 0]}
        right = {"img_path": "a.png", "class_label": 3, "attribute_label": [0, 0]}
        self.assertTrue(followup.same_record_except_label(left, right))
        right["class_label"] = 4
        self.assertFalse(followup.same_record_except_label(left, right))

    def test_residual_means_come_only_from_training_rows(self):
        z_train = np.array([[-2.0], [2.0], [10.0], [14.0]])
        c_train = np.array([[0], [0], [1], [1]])
        z_test = np.array([[100.0], [200.0]])
        c_test = np.array([[0], [1]])
        residual = followup.residualize_from_training(
            z_train, c_train, z_test, c_test
        )
        np.testing.assert_allclose(residual[:, 0], [100.0, 188.0])

    def test_offtarget_evidence_excludes_old_and_inserted_coordinates(self):
        spans = OrderedDict([("tail", (0, 3))])
        swaps = pd.DataFrame(
            {
                "part": ["tail"] * 5,
                "var_src": [0] * 5,
                "var_donor": [1] * 5,
                "sid_src": [0] * 5,
                "sid_donor": [1] * 5,
                "orig_render_id": [f"image-{i}" for i in range(5)],
                "outcome": ["donorward, source wins"] * 5,
                "controlled_event": [True] * 5,
                "m_cf": [-1.0, -0.9, -0.8, -0.7, -0.6],
                "z_cf_tail_0": [999.0] * 5,
                "z_cf_tail_1": [-999.0] * 5,
                "z_cf_tail_2": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        weight = np.zeros((2, 3))
        weight[0, 2] = 2.0
        weight[1, 2] = -1.0
        detail, _ = followup.off_target_saved_head(
            swaps, spans, np.zeros(3), weight
        )
        self.assertEqual(
            detail.offtarget_source_minus_donor_evidence.tolist(),
            [3.0, 6.0, 9.0, 12.0, 15.0],
        )

    def test_image_folds_never_split_an_original(self):
        rows = []
        for species in range(5):
            for image in range(5):
                for repeat in range(2):
                    rows.append(
                        {
                            "orig_render_id": f"s{species}-i{image}",
                            "sid_src": species,
                            "repeat": repeat,
                        }
                    )
        frame = pd.DataFrame(rows)
        folds = followup.source_stratified_folds(frame)
        frame["fold"] = folds
        self.assertEqual(frame.groupby("orig_render_id").fold.nunique().max(), 1)
        self.assertEqual(set(folds), set(range(5)))

    def test_grouped_regression_returns_one_prediction_per_row(self):
        rows = []
        for species in range(5):
            for image in range(5):
                for repeat in range(2):
                    rows.append(
                        {
                            "orig_render_id": f"s{species}-i{image}",
                            "sid_src": species,
                            "part": "tail" if repeat else "wing",
                            "feature": species + repeat / 10,
                            "m_cf": species - repeat,
                        }
                    )
        frame = pd.DataFrame(rows)
        folds = followup.source_stratified_folds(frame)
        prediction = followup.grouped_predictions(
            frame,
            numeric=["feature"],
            categorical=["part"],
            target="m_cf",
            classification=False,
            folds=folds,
        )
        self.assertEqual(len(prediction), len(frame))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
