#!/usr/bin/env python3
"""Static and synthetic checks for the rebuilt official-Koh CUB70 chapter."""
from __future__ import annotations

import json

import build_standard_cbm_reports as reports


def main() -> None:
    notebook = reports.build_cub(preserve_outputs=False)
    cells = notebook["cells"]
    complete_source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"05_cub_cbm.ipynb::code_cell_{index}", "exec")

    required = [
        'CURATED/"koh_joint_resnet_v1"/"cub70"/"standard"/"seed1"',
        "matched_species_diagnostics",
        "matched_species_eligibility",
        "funnybird_species_diagnostics",
        "FB_EVAL,min_each=3,min_positive_fallback=3",
        'E70["pred_label"]=(E70.z>0).astype(int)',
        'assign(_row_index=te.index)',
        'validate="many_to_one"',
        'mask_root.glob(f"{cid}.*")',
        "funnybird_swap_targets",
        "calibrate_recall_warning",
        'CUB70_MODEL_ROOT/"best_model_1.pth"',
        "Figure 4c · Species information available is not the same as saved-head use",
        "mean_probability_mass_moved",
        "for label in np.unique(c_head[te,j])",
        "non-finite CUB70 replacement values",
        "reconstructed CUB70 saved head disagrees with export",
        "Appendix Figure A1",
        "METHOD NOT CALIBRATED AS A BACKWASH PROXY",
        "minimal_cbm: rejected",
    ]
    for text in required:
        assert text in complete_source, text
    forbidden = [
        'CURATED/"cub70_eval"/"cub70-cbm-s1.parquet"',
        "EFULLP",
        "JFULL",
        "same-image full-CUB guard",
        "Task accuracy is 0.1412",
        "The CUB70 model has two exactly collapsed outputs",
        "Held-out raw-z RMSE changes from 3.285",
    ]
    for text in forbidden:
        assert text not in complete_source, text
    assert all(not cell.get("outputs") for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)
    # Ensure the generated object remains valid JSON and the expected figure
    # review slots are present without importing old minimal_cbm numbers.
    json.dumps(notebook)
    for figure in ["1", "2", "3", "4", "4b", "5", "6", "7", "8", "9", "10", "11", "11a", "11b", "12"]:
        assert f"First-pass review slot for Figure {figure}" in complete_source
    print("CUB70 NOTEBOOK 05 BUILDER PASS: official Koh source, recall calibration, and cold-review slots verified")


if __name__ == "__main__":
    main()
