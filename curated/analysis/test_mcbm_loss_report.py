"""Fast synthetic checks for Notebook 03 helpers and generated code.

These checks create no scientific result and perform no model training.
"""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_03_standard_mcbm_report import build
from build_mcbm_parity import TAGS, find
from build_standard_cbm_reports import build_funnybird
from mcbm_loss_report import off_target_erasure, replacement_use, score_reference, softmax, task_head


def synthetic_saved_head(directory: Path):
    torch.manual_seed(3)
    network = torch.nn.Sequential(
        torch.nn.Linear(26, 256), torch.nn.ReLU(), torch.nn.Linear(256, 50)
    )
    checkpoint = directory / "head.pt"
    torch.save(
        {"model": {f"mlp_y.{key}": value for key, value in network.state_dict().items()}},
        checkpoint,
    )
    return task_head(checkpoint), network


def main() -> None:
    print("PREFLIGHT ONLY - SYNTHETIC DATA, NO SCIENTIFIC TRAINING", flush=True)

    standard = build_funnybird()["cells"]
    for tag in TAGS:
        find(standard, tag)

    notebook = build()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for required in (
        "Transition A", "Transition B", "mcbm_swap_pathway_v3",
        "mean_calibrated_h_response", "original_restored_offtarget_summary",
        "NOTEBOOK 03 COMPLETION PASS",
    ):
        assert required in source, required

    curated = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(curated / "data/funnybirds"))
    import funnybirds_concepts
    assert callable(funnybirds_concepts.load_parts)

    rng = np.random.default_rng(241)
    spans = dict(beak=(0, 4), eye=(4, 7), foot=(7, 11), tail=(11, 20), wing=(20, 26))
    names = [f"concept_{index}" for index in range(26)]
    y = np.repeat(np.arange(50), 10)
    c = rng.integers(0, 2, size=(len(y), 26))
    h = (6 * c - 3 + rng.normal(size=c.shape)).astype("float32")

    with tempfile.TemporaryDirectory(prefix="mcbm-report-test-") as temp:
        forward, network = synthetic_saved_head(Path(temp))
        np.testing.assert_allclose(
            forward(h), network(torch.tensor(h)).detach().numpy(), rtol=1e-6, atol=1e-6
        )
        reference = score_reference(h, 1.2 * h, c, names)
        assert len(reference) == 104 and reference.N.min() > 1
        use = replacement_use(h, c, y, softmax(forward(h)), forward, spans)
        assert len(use) == 6 and use.mean_probability_mass_moved.between(0, 1).all()

        rows = []
        for part, (lo, hi) in spans.items():
            for index in range(1000):
                source = index % 50
                old = index % (hi - lo)
                rows.append(dict(
                    part=part, var_src=old, var_donor=(old + 1) % (hi - lo),
                    sid_src=source, sid_donor=(source + 1) % 50,
                    orig_render_id=f"orig-{index % 250:03d}", render_id=f"{part}-{index}",
                    m_cf=0.0, m_orig=-1.0, responded_but_source_wins=False,
                ))
        swaps = pd.DataFrame(rows)
        evidence = off_target_erasure(swaps, np.tile(h, (10, 1)), h, c, forward, spans)
        assert set(evidence.off_target_coordinates) == {1, 2, 4, 7}
        assert evidence.mean_probability_mass_moved.between(0, 1).all()

    print("MCBM NOTEBOOK 03 SYNTHETIC PASS", flush=True)


if __name__ == "__main__":
    main()
