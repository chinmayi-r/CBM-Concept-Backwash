#!/usr/bin/env python3
"""Add stable cell IDs and useful alternative text to notebook 05 outputs.

Run this after execution and before HTML export.  It changes metadata only: code,
markdown, numerical outputs, and plotted pixels are left untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "05_cub_cbm.ipynb"

# A marker from each plotting cell, followed by one description per PNG output.
ALT_BY_MARKER = {
    'concepts=(E70[["concept_name","attribute_type","mask_group"]]': [
        "Bar chart of the number of selected concept values in each of the 28 CUB attribute types."
    ],
    'all_images=E70[["image","y_true"]]': [
        "Bar chart showing masked-image counts for each of the 67 CUB70 species covered by the released mask archive."
    ],
    'fine=(RAWVIS.groupby("part")': [
        "Two-panel chart of visibility for the 11 released masks and the number of visible sides for paired eyes, wings, and legs."
    ],
    'species_concept=(E70.groupby': [
        "Dot plot showing how many CUB70 species carry each exact selected concept value."
    ],
    'EXACT70=exact_visibility_metrics(J70)': [
        "Dot plot of positive-label mask absence and hidden positive prediction rates for every exact concept."
    ],
    'display(task_and_concept_accuracy(E70)': [
        "Two-panel model-health check showing task accuracy and concept-score variation and positive recall by attribute type."
    ],
    'T=(EXACT70.dropna': [
        "Line plot comparing mean positive-concept probability when the mapped mask is absent versus visible."
    ],
    'X=EXACT70.merge': [
        "Scatter plot of label-mask conflict versus hidden positive prediction for exact CUB concepts."
    ],
    'dose=[]\nfor (t,c),d in J70': [
        "Dot plot of the concept-probability difference between the largest and smallest visible-area quartiles."
    ],
    'B=J70[(J70.gt_label==1)': [
        "Line chart of concept probability with zero, one, or two visible eyes, wings, and legs."
    ],
    'MATCH70=matched_effects(J70)': [
        "Dot plot of within-species visible-minus-absent probability effects for exact concepts."
    ],
    'for (t,lab),d in J70.groupby': [
        "Grouped bar chart comparing visibility effects for positive and negative labels by attribute type."
    ],
    'print("Native test populations': [
        "Two-panel comparison of CUB70-trained and full-CUB-trained CBMs, showing hidden violations and score spread.",
        "Scatter plot comparing species-matched visibility effects in CUB70-trained and full-CUB-trained CBMs.",
    ],
    'conflict_bar = (EXACT70.groupby': [
        "Grouped bar chart of CUB label-mask conflict and hidden positive prediction by attribute type."
    ],
    'def species_probe(frame, block, value)': [
        "Grouped bar chart of held-out species accuracy decoded from true concept labels and from CBM probabilities for each mask block."
    ],
    'eligible = EXACT70[': [
        "Four-row diagnostic grid of real CUB photographs and every available part-mask overlay for selected conflict and visibility cases."
    ],
    'POS["mask_state"] = np.select': [
        "Stacked bar chart separating zero-pixel, tiny-mask, and visible-mask positive-labelled examples by attribute type."
    ],
    'contrast_rows = []': [
        "Box plots of within-species visible-minus-absent change in correct-versus-best-wrong concept contrast."
    ],
}


def stable_id(index: int, source: str) -> str:
    digest = hashlib.sha1(f"cub05:{index}:{source}".encode("utf-8")).hexdigest()[:12]
    return f"cub05-{digest}"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    used: set[str] = set()
    image_count = 0

    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        cell_id = cell.get("id")
        if not cell_id or cell_id in used:
            cell_id = stable_id(index, source)
            cell["id"] = cell_id
        used.add(cell_id)

        image_outputs = [
            output for output in cell.get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        if not image_outputs:
            continue

        matches = [alts for marker, alts in ALT_BY_MARKER.items() if marker in source]
        if len(matches) != 1:
            raise RuntimeError(
                f"plot cell {index} has {len(matches)} alternative-text matches; "
                f"source starts {source[:100]!r}"
            )
        alts = matches[0]
        if len(alts) != len(image_outputs):
            raise RuntimeError(
                f"plot cell {index} produced {len(image_outputs)} images but has "
                f"{len(alts)} descriptions"
            )

        # The classic nbconvert template reads cell.metadata.alt.  Output-level
        # metadata is also populated so future/custom templates can distinguish
        # multiple figures emitted by one cell.
        cell.setdefault("metadata", {})["alt"] = alts[0]
        for output, alt in zip(image_outputs, alts):
            metadata = output.setdefault("metadata", {})
            metadata["alt"] = alt
            metadata.setdefault("image/png", {})["alt"] = alt
            image_count += 1

    missing_ids = [i for i, cell in enumerate(notebook["cells"]) if not cell.get("id")]
    if missing_ids:
        raise RuntimeError(f"cells still lack IDs: {missing_ids}")
    if image_count != 19:
        raise RuntimeError(f"expected 19 executed PNG outputs, found {image_count}")

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"[NOTEBOOK METADATA PASS] {len(notebook['cells'])} cells have IDs; "
        f"{image_count} plotted images have descriptions"
    )


if __name__ == "__main__":
    main()
