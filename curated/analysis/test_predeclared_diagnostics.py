"""Focused synthetic tests for D6 scientific invariants; no research results."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import diag_common as dc
import diag_dimension_adjusted_information as d61
import diag_grouped_risk_model as d64
import diag_profile_transfer as d62


class DiagnosticInvariantTests(unittest.TestCase):
    def test_controlled_event_excludes_zero_final_margin(self):
        frame = pd.DataFrame({
            "response_delta": [1.0, 1.0, -1.0],
            "m_cf": [-0.1, 0.0, -0.1],
        })
        self.assertEqual(dc.controlled_event(frame).tolist(), [True, False, False])

    def test_label_residual_uses_training_means(self):
        z_train = np.array([[1.0], [3.0], [10.0], [14.0]])
        c_train = np.array([[0], [0], [1], [1]])
        z_test = np.array([[4.0], [16.0]])
        c_test = np.array([[0], [1]])
        train, test = d61._within_label_residual(z_train, c_train, z_test, c_test)
        np.testing.assert_allclose(train.ravel(), [-1.0, 1.0, -2.0, 2.0])
        np.testing.assert_allclose(test.ravel(), [2.0, 4.0])

    def test_conditional_probe_detects_species_signal_beyond_labels(self):
        rng = np.random.default_rng(10)
        species = np.repeat(np.arange(10), 10)
        labels = np.zeros((len(species), 10), dtype=int)
        labels[np.arange(len(species)), species % 2] = 1
        raw = labels.astype(float) * 4.0
        raw += np.eye(10)[species] * 2.0
        raw += rng.normal(0, .03, raw.shape)
        result = d61.summarize(raw, labels, species, with_interval=False)
        self.assertGreater(result["conditional_logloss_gain"], 0.5)

    def test_off_target_eye_is_not_interpretable_with_one_coordinate(self):
        rows = []
        for part, coordinates in [("eye", 1), ("tail", 7)]:
            for index in range(25):
                rows.append({
                    "stage": "post", "variant": "off_target", "part": part,
                    "outcome": "donor wins", "original_image": f"i{index}",
                    "coordinates_used": coordinates,
                    "donor_similarity": .5, "source_similarity": .1,
                    "relative_transfer": .4,
                })
        summary = d62.summarize(pd.DataFrame(rows))
        self.assertFalse(bool(summary.loc[summary.part == "eye", "interpretable"].iloc[0]))
        self.assertTrue(bool(summary.loc[summary.part == "tail", "interpretable"].iloc[0]))

    def test_strict_transport_features_do_not_include_alternatives(self):
        self.assertNotIn("alternatives_in_part", d64.CORE)
        self.assertIn("alternatives_in_part", d64.STRUCTURAL)

    def test_nested_ridge_and_logistic_return_finite_predictions(self):
        rng = np.random.default_rng(4)
        groups = np.repeat([f"image_{i}" for i in range(24)], 3)
        X = rng.normal(size=(len(groups), 4))
        y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, .1, len(groups))
        event = (y < np.median(y)).astype(int)
        ridge_prediction, ridge_alpha, _ = d64.fit_ridge(
            X[:54], y[:54], X[54:], groups[:54])
        event_prediction, event_alpha = d64.fit_logistic(
            X[:54], event[:54], X[54:], groups[:54])
        self.assertTrue(np.isfinite(ridge_prediction).all())
        self.assertTrue(np.isfinite(event_prediction).all())
        self.assertIn(ridge_alpha, d64.ALPHAS)
        self.assertIn(event_alpha, d64.ALPHAS)

    def test_record_identity_detects_reordering(self):
        first = {"image": "a/x.png", "class_label": 1, "id": 4}
        second = {"image": "b/x.png", "class_label": 1, "id": 4}
        self.assertNotEqual(dc._record_identity(first), dc._record_identity(second))


if __name__ == "__main__":
    unittest.main()
