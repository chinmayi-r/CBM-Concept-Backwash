#!/usr/bin/env python3
"""Static/synthetic contract checks for notebooks 05, 06, and 07."""
from __future__ import annotations
import ast
import json
from pathlib import Path

CURATED=Path(__file__).resolve().parents[1]

def text(path: Path) -> str:
    nb=json.loads(path.read_text(encoding="utf-8"))
    for item in nb["cells"]:
        if item["cell_type"]=="code":
            ast.parse("".join(item["source"]))
    return "\n".join("".join(item["source"]) for item in nb["cells"])

def main() -> None:
    n05=text(CURATED/"notebooks/05_cub_cbm.ipynb")
    assert "koh_joint_resnet_v1" in n05
    assert "cub70_eval/cub70-cbm" not in n05
    assert "JFULL" not in n05 and "EFULL" not in n05
    assert "concept-specific Grad-CAM" in n05
    assert "CUB70_HEAD_USE" in n05

    n06=text(CURATED/"notebooks/06_cub_mcbm.ipynb")
    for token in ('0.:"0"', '.1:"0p1"', '.3:"0p3"', '1.:"1"'):
        assert token in n06
    assert "Koh Standard → MCBM gamma 0" in n06
    assert "MCBM gamma 0 → positive gamma" in n06
    assert "There is no accepted CUB donor-part swap" in n06

    n07=text(CURATED/"notebooks/07_full_cub_cbm.ipynb")
    assert "koh_joint_resnet_decay_continuation_v1" in n07
    assert "Complete entry 11 before rendering Full CUB" in n07
    assert "saved Wz+b" in n07
    assert "concept-specific gradients" in n07
    assert "legacy `minimal_cbm` CBM" in n07
    print("CUB NOTEBOOK 05/06/07 CONTRACT PASS")

if __name__=="__main__":
    main()
