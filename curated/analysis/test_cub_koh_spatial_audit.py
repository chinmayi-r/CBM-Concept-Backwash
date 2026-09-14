#!/usr/bin/env python3
"""Behavioral tests for the CUB Koh spatial audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from cub_koh_spatial_audit import (
    audit_visibility_artifact,
    choose_candidates,
    cross_fitted_label_means,
    gradcam_maps,
    koh_center_crop_mask,
    koh_eval_loader_kwargs,
    koh_pkl_paths,
    localization_metrics,
    native_coarse_mask,
    released_mask_index,
    saved_head_use_table,
    summarize_replay,
    validate_candidate_geometry,
)


def test_real_mask_identity_crop_and_candidate_contract() -> None:
    with TemporaryDirectory() as temp:
        work = Path(temp)
        root = work / "masks" / "AnnotationMasksPerclass"
        dotted = root / "9.California_Gull"
        dotted.mkdir(parents=True)
        from PIL import Image
        inside_stem = "California_Gull_0091_41276"
        outside_stem = "California_Gull_0092_41277"
        inside = np.zeros((400, 400), dtype=np.uint8); inside[100:110, 100:110] = 255
        tiny_stem = "California_Gull_0093_41278"
        tiny = np.zeros((400, 400), dtype=np.uint8); tiny[100:102, 100:102] = 255
        outside = np.zeros((400, 400), dtype=np.uint8); outside[5:7, 5:7] = 255
        Image.fromarray(inside).save(dotted / f"{inside_stem}_left_eye.png")
        Image.fromarray(tiny).save(dotted / f"{tiny_stem}_left_eye.png")
        Image.fromarray(outside).save(dotted / f"{outside_stem}_left_eye.png")
        # A misleading numeric directory is deliberately present.  The old
        # implementation chose it from the model class label and missed the mask.
        (root / "9").mkdir()
        index = released_mask_index(work / "masks")
        native = native_coarse_mask(index, inside_stem, "eye")
        assert native is not None and native.shape == (400, 400)
        cropped = koh_center_crop_mask(native)
        assert cropped.shape == (299, 299) and cropped.mean() >= 0.001

        visibility = pd.DataFrame([
            {"image_name": stem, "part": "left_eye",
             "pixel_count": 100 if stem == inside_stem else 4,
             "img_pixels": 160000}
            for stem in (inside_stem, tiny_stem, outside_stem)
        ])
        artifact = audit_visibility_artifact(visibility, index)
        assert artifact["indexed_fine_masks"] == 3
        evaluation = pd.DataFrame([
            {"image": stem, "y_true": 8, "concept_index": 0,
             "concept_name": "has_eye_color::blue", "gt_label": 1, "z": 2.0}
            for stem in (inside_stem, tiny_stem, outside_stem)
        ])
        candidates, model_view, masks = choose_candidates(evaluation, index, per_group=48)
        assert candidates.image.tolist() == [inside_stem]
        view = model_view.set_index("image")
        assert view.loc[inside_stem, "model_mask_visible"]
        assert view.loc[tiny_stem, "model_mask_nonempty"]
        assert not view.loc[tiny_stem, "model_mask_visible"]
        assert not view.loc[outside_stem, "model_mask_nonempty"]

        image_dir = work / "CUB_200_2011" / "images" / "009.California_Gull"
        image_dir.mkdir(parents=True)
        records = []
        for stem in (inside_stem, tiny_stem, outside_stem):
            path = image_dir / f"{stem}.jpg"
            Image.new("RGB", (400, 400), "white").save(path)
            records.append({"img_path": str(path), "class_label": 8,
                            "attribute_label": [1]})
        geometry = validate_candidate_geometry(candidates, masks, records, work, index)
        assert geometry["minimum_model_mask_pixels"] == 100
        assert geometry["minimum_selected_model_mask_area_fraction"] >= 0.001


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
        "resol": 299,
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


def test_gradcam_map_formula_and_shape() -> None:
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]])
    gradients = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]])
    positive, absolute = gradcam_maps(features, gradients, (4, 4))
    assert positive.shape == absolute.shape == (4, 4)
    assert np.isfinite(positive).all() and np.isfinite(absolute).all()
    assert (positive >= 0).all() and (absolute >= 0).all()


def test_tiny_nonempty_center_cropped_mask_is_valid() -> None:
    """A mask below 0.1% is small, not empty, on the model grid."""
    mask = np.zeros((299, 299), dtype=bool)
    mask[100:102, 100:102] = True
    assert 0 < mask.mean() < 0.001
    heatmap = mask.astype(float)
    result = localization_metrics(heatmap, mask)
    assert result["mask_area_fraction"] == mask.mean()
    assert result["attribution_mass_inside_mask"] == 1.0


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
    test_real_mask_identity_crop_and_candidate_contract()
    test_koh_loader_receives_string_paths()
    test_koh_loader_matches_recorded_export_contract()
    test_replay_audit_accepts_small_cuda_noise_and_rejects_real_drift()
    test_localization_metrics()
    test_gradcam_map_formula_and_shape()
    test_tiny_nonempty_center_cropped_mask_is_valid()
    test_cross_fitted_means_do_not_use_held_out_rows()
    test_saved_head_use_detects_used_magnitude()
    print("CUB KOH SPATIAL AUDIT SYNTHETIC PASS")
