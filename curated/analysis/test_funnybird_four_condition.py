#!/usr/bin/env python3
"""Behavioral tests for the FunnyBird four-condition diagnostic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Import without executing CLI main.
from funnybird_four_condition_core import (  # noqa: E402
    add_contrasts, delete_part, exact_target_composite, four_conditions,
    image_rgb_mae, mask_iou, summarize, target_rgb_mae,
)


def test_annotations() -> None:
    ann = {
        "class_idx": 3,
        "tail_model": "tail01.glb", "tail_color": "blue",
        "wing_model": "wing02.glb", "wing_color": "red",
        "eye_model": "eye01.glb", "beak_model": "beak01.glb",
        "beak_color": "yellow", "foot_model": "foot01.glb",
        "camera_pitch": 4, "bg_objects": "cube", "bg_color": "green",
    }
    parts = ["beak", "eye", "wing", "foot", "tail"]
    colored = {"tail", "wing", "beak"}
    conditions = four_conditions(ann, "tail", parts, colored)
    assert conditions["11"] == ann
    assert conditions["01"]["tail_model"] == ""
    assert conditions["01"]["tail_color"] == ""
    assert conditions["01"]["wing_model"] == ann["wing_model"]
    assert conditions["10"]["tail_model"] == ann["tail_model"]
    assert conditions["10"]["wing_model"] == ""
    assert conditions["10"]["camera_pitch"] == ann["camera_pitch"]
    assert conditions["10"]["bg_objects"] == ann["bg_objects"]
    assert all(conditions["00"][f"{part}_model"] == "" for part in parts)
    assert delete_part(ann, "tail", colored)["class_idx"] == 3


def test_contrasts_and_closure() -> None:
    # Concrete example from the report: 6, 4, 1, -3.
    frame = add_contrasts(pd.DataFrame({
        "z11": [6.0], "z01": [4.0], "z10": [1.0], "z00": [-3.0],
    }))
    row = frame.iloc[0]
    assert row.part_response_with_context == 2.0
    assert row.part_response_without_context == 4.0
    assert row.context_evidence_with_part == 5.0
    assert row.context_evidence_without_part == 7.0
    assert row.interaction == -2.0
    frame["part"] = "tail"
    frame["eligible"] = False
    summary = summarize(frame)
    assert summary.loc[0, "selected_rows"] == 1
    assert summary.loc[0, "eligible_rows"] == 0
    assert np.isnan(summary.loc[0, "median_z11"])


def test_masks_and_rgb() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[1:3, 1:3] = True
    b[1:3, 1:3] = True
    assert mask_iou(a, b) == 1.0
    from PIL import Image
    x = np.zeros((4, 4, 3), dtype=np.uint8)
    y = x.copy()
    y[a] = 4
    assert target_rgb_mae(Image.fromarray(x), Image.fromarray(y), a, b) == 4.0
    source = np.full((4, 4, 3), 9, dtype=np.uint8)
    background = np.full((4, 4, 3), 2, dtype=np.uint8)
    composite = np.asarray(exact_target_composite(
        Image.fromarray(source), Image.fromarray(background), a))
    assert np.all(composite[a] == 9)
    assert np.all(composite[~a] == 2)
    assert image_rgb_mae(Image.fromarray(x), Image.fromarray(y)) == 1.0


def test_symmetric_pixel_factorial() -> None:
    """Both part-present cells must use the same exact-pixel insertion."""
    from PIL import Image
    target = np.zeros((4, 4), dtype=bool)
    target[1:3, 1:3] = True
    native_11 = np.full((4, 4, 3), 5, dtype=np.uint8)
    native_11[target] = 9
    base_01 = np.full((4, 4, 3), 5, dtype=np.uint8)
    base_00 = np.full((4, 4, 3), 2, dtype=np.uint8)
    scientific_11 = np.asarray(exact_target_composite(
        Image.fromarray(native_11), Image.fromarray(base_01), target))
    scientific_10 = np.asarray(exact_target_composite(
        Image.fromarray(native_11), Image.fromarray(base_00), target))
    assert np.array_equal(scientific_11[target], scientific_10[target])
    assert np.all(scientific_11[~target] == 5)
    assert np.all(scientific_10[~target] == 2)


if __name__ == "__main__":
    test_annotations()
    test_contrasts_and_closure()
    test_masks_and_rgb()
    test_symmetric_pixel_factorial()
    print("FUNNYBIRD FOUR-CONDITION SYNTHETIC PASS")
