#!/usr/bin/env python3
"""Behavioral tests for the CUB Koh spatial audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from cub_koh_spatial_audit import (
    coarse_mask,
    cross_fitted_label_means,
    koh_eval_loader_kwargs,
    koh_pkl_paths,
    localization_metrics,
    released_mask_index,
    saved_head_use_table,
    summarize_replay,
    validate_candidate_masks,
)


def test_mask_index_uses_actual_archive_filenames_not_model_class_ids() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp) / "AnnotationMasksPerclass"
        dotted = root / "9.California_Gull"
        dotted.mkdir(parents=True)
        from PIL import Image
        Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(
            dotted / "California_Gull_0091_41276_left_eye.png"
        )
        # A misleading numeric directory is deliberately present.  The old
        # implementation chose it from the model class label and missed the mask.
        (root / "9").mkdir()
        index = released_mask_index(Path(temp))
        candidates = pd.DataFrame([{
            "image": "California_Gull_0091_41276", "mask_group": "eye"
        }])
        audit = validate_candidate_masks(candidates, index)
        assert audit["selected_image_group_pairs"] == 1
        mask = coarse_mask(index, "California_Gull_0091_41276", "eye", (4, 4))
        assert mask.all()


def test_koh_loader_receives_string_paths() -> None:
    """Guard Koh's real ``'train.pkl' in path`` interface assumption."""
    data_pkl = Path("selection/test.pkl")
    paths = koh_pkl_paths(data_pkl)
    assert all(isinstance(path, str) for path in paths)
    assert "train.pkl" not in paths[0]
    assert "test.pkl" in paths[0]


def test_koh_loader_matches_recorded_export_contract() -> None:
    assert koh_eval_loader_kwargs() == {
        "use_attr": True,
        "no_img": False,
        "batch_size": 64,
        "uncertain_label": False,
        "n_class_attr": 2,
        "image_dir": "images",
        "resampling": False,
    }


def test_replay_audit_accepts_small_cuda_noise_and_rejects_real_drift() -> None:
    expected = np.array([[-2.0, -0.001, 3.0], [1.0, 2.0, -4.0]])
    replayed = expected + np.array([[0.001, 0.0015, -0.002], [0.0, 0.003, -0.001]])
    small = summarize_replay(expected, replayed, np.array([0, 1]), np.array([0, 1]))
    assert small["accepted"]
    assert small["strict_pass"]
    assert small["concept_sign_changes"] == 1
    assert small["concept_sign_changes_outside_strict_boundary"] == 0
    disclosed = expected.copy()
    disclosed[0, 2] += 0.03
    middle = summarize_replay(expected, disclosed, np.array([0, 1]), np.array([0, 1]))
    assert middle["accepted"]
    assert not middle["strict_pass"]
    assert middle["acceptance_mode"] == "disclosed_historical_cuda_difference"
    drifted = replayed.copy()
    drifted[0, 0] = 1.0
    large = summarize_replay(expected, drifted, np.array([0, 1]), np.array([0, 1]))
    assert not large["accepted"]
    assert large["concept_sign_changes_outside_strict_boundary"] == 1


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
    test_mask_index_uses_actual_archive_filenames_not_model_class_ids()
    test_koh_loader_receives_string_paths()
    test_koh_loader_matches_recorded_export_contract()
    test_replay_audit_accepts_small_cuda_noise_and_rejects_real_drift()
    test_localization_metrics()
    test_cross_fitted_means_do_not_use_held_out_rows()
    test_saved_head_use_detects_used_magnitude()
    print("CUB KOH SPATIAL AUDIT SYNTHETIC PASS")
