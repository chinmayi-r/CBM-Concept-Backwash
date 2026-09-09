"""Synthetic tests for the h-to-z controlled-swap pathway arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcbm_swap_pathway_report import pathway_rows, summarize


def test_pathway_separates_encoder_movement_from_q_damage():
    spans = {"tail": (0, 3)}
    swaps = pd.DataFrame([
        dict(gamma=5.0, part="tail", render_id="a", orig_render_id="o1", var_src=0, var_donor=1),
        dict(gamma=5.0, part="tail", render_id="b", orig_render_id="o2", var_src=0, var_donor=1),
    ])
    # In both rows the encoder makes the inserted donor largest in h.
    h_orig = np.array([[3, -3, -3], [3, -3, -3]], dtype=float)
    h_cf = np.array([[-2, 2, -3], [-2, 2, -3]], dtype=float)
    # q preserves that result once, but reverses it once.
    z_orig = np.array([[5, -5, -5], [5, -5, -5]], dtype=float)
    z_cf = np.array([[-4, 4, -5], [4, -4, -5]], dtype=float)
    rows = pathway_rows(
        swaps, h_orig, h_cf, z_orig, z_cf, spans, np.array([True, False])
    )
    assert np.allclose(rows.h_response, 10)
    assert rows.h_exact_donor_recognized.tolist() == [True, True]
    assert rows.z_exact_donor_recognized.tolist() == [True, False]
    assert rows.q_breaks_h_exact_success.tolist() == [False, True]
    assert np.allclose(rows.z_response, [18, 2])
    summary = summarize(rows)
    all_rows = summary[summary.population.eq("all rows")].iloc[0]
    strict = summary[summary.population.eq("strict matched replay")].iloc[0]
    assert all_rows.n_rows == 2 and strict.n_rows == 1
    assert np.isclose(all_rows.q_breaks_h_success_rate, 0.5)
    assert np.isclose(strict.z_exact_donor_recognition, 1.0)


if __name__ == "__main__":
    test_pathway_separates_encoder_movement_from_q_damage()
    print("MCBM SWAP PATHWAY SYNTHETIC PASS")
