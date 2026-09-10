#!/usr/bin/env python3
"""Source-only checks for the Notebook 03 -> 03rl parity contract."""
from __future__ import annotations

import ast

from build_03_standard_mcbm_report import build as build_standard
from build_03rl_notebook import STANDARD_OUTPUT_PREFIXES, build as build_rlv2


def prefix(cell: dict) -> str:
    identity = cell.get("id", "")
    stem = "m3rl-" if identity.startswith("m3rl-") else "m3-"
    if not identity.startswith(stem):
        return ""
    return identity[len(stem):].rsplit("-", 1)[0]


def test_rlv2_is_complete_standard_template_substitution() -> None:
    standard = build_standard()
    rlv2 = build_rlv2()
    for notebook in (standard, rlv2):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    text = "\n".join("".join(cell.get("source", [])) for cell in rlv2["cells"])
    for forbidden in (
        "swap_fixed_v2_attempt2", "mcbm_swap_pathway_v3",
        "mcbm_notebook03_tables", "funnybirds-mcbm-{TAG[g]}",
    ):
        assert forbidden not in text, forbidden
    for required in (
        "swap_fixed_v3_matched", "mcbm_swap_pathway_rlv2_v3",
        "mcbm_notebook03rl_tables", "funnybirds-mcbm-rlv2matched-{TAG[g]}",
        "mcbm_notebook03rl_standard_difference_all_fronts.csv",
    ):
        assert required in text, required

    identities = [prefix(cell) for cell in rlv2["cells"]]
    for output_prefix in STANDARD_OUTPUT_PREFIXES:
        reference = f"standard-{output_prefix}"
        assert identities.count(reference) == 1, reference
        assert identities.count(output_prefix) == 1, output_prefix
        assert identities.index(reference) < identities.index(output_prefix)


if __name__ == "__main__":
    test_rlv2_is_complete_standard_template_substitution()
    print("NOTEBOOK 03RL PARITY SOURCE PASS")
