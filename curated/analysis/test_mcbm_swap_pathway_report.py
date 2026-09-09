"""Synthetic tests for the h-to-z controlled-swap pathway arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mcbm_swap_pathway_report import (
    attach_measured_contributors,
    original_restored_offtarget_intervention,
    pathway_rows,
    summarize,
    summarize_hybrid,
    summarize_values,
)


def test_pathway_separates_encoder_movement_from_q_damage():
    # Match the real FunnyBird representation width.  Only the first three
    # coordinates are used by this small tail example; the remaining 23 are
    # unchanged sentinels.
    spans = {"tail": (0, 3)}
    swaps = pd.DataFrame([
        dict(gamma=5.0, part="tail", render_id="a", orig_render_id="o1", var_src=0, var_donor=1),
        dict(gamma=5.0, part="tail", render_id="b", orig_render_id="o2", var_src=0, var_donor=1),
    ])
    # In both rows the encoder makes the inserted donor largest in h.
    h_orig = np.full((2, 26), -3.0)
    h_cf = np.full((2, 26), -3.0)
    h_orig[:, :3] = [[3, -3, -3], [3, -3, -3]]
    h_cf[:, :3] = [[-2, 2, -3], [-2, 2, -3]]
    # q preserves that result once, but reverses it once.
    z_orig = np.full((2, 26), -5.0)
    z_cf = np.full((2, 26), -5.0)
    z_orig[:, :3] = [[5, -5, -5], [5, -5, -5]]
    z_cf[:, :3] = [[-4, 4, -5], [4, -4, -5]]
    absent = np.full(26, -3.0)
    present = np.full(26, 3.0)
    rows = pathway_rows(
        swaps, h_orig, h_cf, z_orig, z_cf, spans, np.array([True, False]),
        absent, present,
    )
    assert np.allclose(rows.h_response, 10)
    assert rows.h_exact_donor_recognized.tolist() == [True, True]
    assert rows.z_exact_donor_recognized.tolist() == [True, False]
    assert rows.h_exact_valid.tolist() == [True, True]
    assert rows.h_exact_winner_value.tolist() == [1, 1]
    assert rows.z_exact_winner_value.tolist() == [1, 0]
    assert rows.z_exact_source_recognized.tolist() == [False, True]
    assert rows.z_exact_third_value_wins.tolist() == [False, False]
    assert rows.q_breaks_h_exact_success.tolist() == [False, True]
    assert np.allclose(rows.z_response, [18, 2])
    summary = summarize(rows)
    all_rows = summary[summary.population.eq("all rows")].iloc[0]
    strict = summary[summary.population.eq("strict matched replay")].iloc[0]
    assert all_rows.n_rows == 2 and strict.n_rows == 1
    assert np.isclose(all_rows.q_breaks_h_success_rate, 0.5)
    assert np.isclose(strict.z_exact_donor_recognition, 1.0)


def test_original_restored_offtarget_intervention_only_restores_third_slot():
    spans = {"tail": (0, 3)}
    swaps = pd.DataFrame([
        dict(render_id="a", orig_render_id="o1", part="tail", var_src=0,
             var_donor=1, sid_src=0, sid_donor=1),
    ])
    h_orig = np.zeros((1, 26))
    h_cf = np.zeros((1, 26))
    h_orig[0, :3] = [3, -3, -2]
    h_cf[0, :3] = [-3, 3, 5]

    def forward(h):
        logits = np.zeros((len(h), 2))
        logits[:, 0] = h[:, 2]
        logits[:, 1] = h[:, 1]
        return logits

    result = original_restored_offtarget_intervention(
        swaps, h_orig, h_cf, spans, forward
    )
    assert result.restored_offtarget_coordinates.tolist() == [1]
    assert np.isclose(result.source_minus_donor_gap_before.iloc[0], 2)
    assert np.isclose(result.source_minus_donor_gap_after.iloc[0], -5)
    assert np.isclose(result.swap_induced_offtarget_source_evidence.iloc[0], 7)
    assert result.pairwise_source_to_donor_flip.tolist() == [True]
    summary = summarize_hybrid(result.assign(gamma=5.0))
    assert np.isclose(summary.mean_swap_induced_offtarget_source_evidence.iloc[0], 7)


def test_measured_contributors_join_by_render_part_and_exact_value():
    rows = pd.DataFrame([
        dict(
            gamma=5.0, render_id="r", original_image="o", part="tail",
            source_value=0, donor_value=1, z_original_margin=-8.0,
            z_donor_gain=3.0, z_source_decrease=2.0, z_response=5.0,
            z_final_margin=-3.0, z_no_donorward_movement=False,
            z_responded_but_source_above_donor=True,
            z_exact_donor_recognized=False, z_exact_source_recognized=True,
            z_exact_third_value_wins=False, h_exact_valid=True,
            h_exact_donor_recognized=True,
            h_exact_donor_recognized_valid=1.0,
        )
    ])
    swaps = pd.DataFrame([
        dict(render_id="r", part="tail", direction="fwd", sid_src=2,
             sid_donor=3, pixel_count_cf=11)
    ])
    visibility = pd.DataFrame([
        dict(render_id="r", part="tail", corrected_all_instance_pixels=22)
    ])
    factors = pd.DataFrame([
        dict(part="tail", value=0, conflict_rate=.2, species_support=4,
             positive_images=300),
        dict(part="tail", value=1, conflict_rate=.3, species_support=5,
             positive_images=400),
    ])
    joined = attach_measured_contributors(rows, swaps, visibility, factors)
    assert joined.corrected_visible_pixels.tolist() == [22]
    assert joined.source_species_support.tolist() == [4]
    assert joined.donor_species_support.tolist() == [5]
    summary = summarize_values(joined)
    assert summary.n_rows.tolist() == [1]
    assert summary.responded_but_source_above_donor_rate.tolist() == [1.0]


if __name__ == "__main__":
    test_pathway_separates_encoder_movement_from_q_damage()
    test_original_restored_offtarget_intervention_only_restores_third_slot()
    test_measured_contributors_join_by_render_part_and_exact_value()
    print("MCBM SWAP PATHWAY SYNTHETIC PASS")
