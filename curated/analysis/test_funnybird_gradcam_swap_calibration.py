#!/usr/bin/env python3
"""Behavioral tests for the FunnyBird Grad-CAM/swap bridge."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from funnybird_gradcam_swap_calibration import (  # noqa: E402
    PARTS, build_examples, center_crop_array, changed_pixel_mask, model_tensor,
    run_gradcam, select_rows, summarize,
)
from PIL import Image  # noqa: E402


def test_center_crop_and_padding() -> None:
    large = np.zeros((401, 501), dtype=bool); large[200, 250] = True
    cropped = center_crop_array(large, 299)
    assert cropped.shape == (299, 299) and cropped[149, 149]
    small = np.ones((5, 7), dtype=bool)
    padded = center_crop_array(small, 9)
    assert padded.shape == (9, 9) and padded.sum() == 35


def test_changed_pixels() -> None:
    a = np.zeros((299, 299, 3), dtype=np.uint8)
    b = a.copy(); b[100:110, 80:90] = 255
    mask = changed_pixel_mask(Image.fromarray(a), Image.fromarray(b))
    assert mask.shape == (299, 299) and mask.sum() == 100


def test_selection_and_summary() -> None:
    rows = []
    for part in PARTS:
        for i in range(30):
            response = float(i - 10)
            margin = -1.0 if i % 2 == 0 else 1.0
            rows.append({"part": part, "render_id": f"{part}-{i}",
                         "orig_render_id": f"o-{i % 7}", "response_delta": response,
                         "margin": margin, "var_donor": i % 3})
    selected = select_rows(pd.DataFrame(rows), 10, 7)
    assert selected.groupby("part").size().eq(10).all()
    selected["donor_minus_source_margin_positive_area_adjusted_enrichment"] = np.arange(len(selected))
    selected["changed_pixel_fraction"] = 0.1
    summary = summarize(selected)
    assert summary.part.tolist() == list(PARTS)
    assert summary.n_rows.eq(10).all()


class _First(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Conv2d(3, 4, kernel_size=1)


class _FakeKoh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_model = _First()

    def forward(self, values):
        features = self.first_model.layer4(values)
        pooled = features.mean(dim=(2, 3))
        concepts = []
        for index in range(26):
            concepts.append((pooled[:, index % 4] + index / 10).reshape(-1, 1))
        task = torch.zeros((values.shape[0], 50), device=values.device)
        return [task, *concepts]


def test_gradcam_chain_and_example_render() -> None:
    spans = {"beak": (0, 4), "eye": (4, 7), "foot": (7, 11),
             "tail": (11, 20), "wing": (20, 26)}
    model = _FakeKoh().eval()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for number, part in enumerate(PARTS):
            a = np.zeros((299, 299, 3), dtype=np.uint8) + 80
            b = a.copy(); b[40 + number:60 + number, 70:95] = 180
            original, counterfactual = Image.fromarray(a), Image.fromarray(b)
            op, cp = root / f"{part}-o.png", root / f"{part}-c.png"
            original.save(op); counterfactual.save(cp)
            with torch.no_grad():
                z = torch.cat(model(model_tensor(counterfactual).unsqueeze(0))[1:], dim=1)[0]
            lo, _ = spans[part]
            rows.append({"render_id": f"r-{part}", "orig_render_id": f"o-{part}",
                         "part": part, "var_src": 0, "var_donor": 1,
                         "image_orig_path": str(op), "image_cf_path": str(cp),
                         "z_old": float(z[lo]), "z_new": float(z[lo + 1]),
                         "margin": 0.1, "response_delta": 0.2,
                         "controlled_event": False})
        metrics, extremes = run_gradcam(model, pd.DataFrame(rows), torch.device("cpu"), spans)
        assert len(metrics) == 5 and set(metrics.part) == set(PARTS)
        output = root / "examples.png"
        build_examples(extremes, output)
        assert output.is_file() and output.stat().st_size > 0


if __name__ == "__main__":
    test_center_crop_and_padding()
    test_changed_pixels()
    test_selection_and_summary()
    test_gradcam_chain_and_example_render()
    print("FUNNYBIRD GRADCAM/SWAP CALIBRATION SYNTHETIC PASS")
