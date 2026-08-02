#!/usr/bin/env python3
"""Append the identical randomized-patch evidence chain to notebooks 02 and 05."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

CURATED = Path(__file__).resolve().parents[1]
NOTEBOOKS = [CURATED / "notebooks" / "02_funnybirds_cbm.ipynb",
             CURATED / "notebooks" / "05_cub_cbm.ipynb"]
SECTION = "fb_cub70_randomized_patch_v1"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def cell_id(name: str) -> str:
    return "rpatch-" + hashlib.sha1(name.encode()).hexdigest()[:12]


def md(text: str, name: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id(name),
            "metadata": {"paired_section": SECTION, "paired_cell": name},
            "source": lines(text)}


def code(text: str, name: str, alt: str) -> dict:
    return {"cell_type": "code", "id": cell_id(name), "execution_count": None,
            "outputs": [], "metadata": {"paired_section": SECTION,
                                           "paired_cell": name, "alt": alt},
            "source": lines(text)}


CELLS = [
    md(r"""
    ## Robustness follow-up: small smooth randomized masks

    The previous whole-part inpainting test failed its FunnyBird calibration. Its
    large edited regions and meaningful-part controls made the CUB70 result
    ambiguous. We therefore ask a smaller question:

    **Does repeatedly covering small local regions inside part `j` change its exact
    concept score `z_j` more than covering the same amount elsewhere?**

    For each visible positive concept we use four increasing doses and four random
    placements. At every dose we compare patches:

    1. inside the named target part;
    2. as an exact translated copy on other bird pixels, away from that part;
    3. as another exact translated copy on background pixels.

    We repeat every mask with two fills: a local blur and a local-neighbour mean
    colour. Agreement across fills makes a result less likely to be caused by one
    particular replacement texture. The masks have soft Gaussian edges and never
    intentionally cover the whole part.
    """, "header"),

    md(r"""
    ### Variables and predictions

    - `delta_z = z_masked - z_original`. Negative means masking lowered the exact
      concept's raw internal score.
    - `adjusted_drop_p = (p_original-p_target) - (p_original-p_other_bird)`.
      Positive means target patches hurt more than equally sized patches elsewhere
      on the bird.
    - `target_retained_p = p_target_masked / p_original`. A value near one means
      most of the original concept probability remains at that partial masking dose.

    **Local-pixel-reliance prediction:** as target dose rises, `delta_z` should fall
    more strongly for target patches than for either control, under both fills.

    **Context-retention prediction:** target patches can have a real dose effect while
    `target_retained_p` remains high. This means the tested local pixels matter but
    partial removal has not eliminated the concept answer. It does **not** prove that
    species supplied the remainder; the exact-concept species residual remains a
    separate observational clue.
    """, "variables"),

    md(r"""
    ### Gate 1: FunnyBird must work before CUB70 is allowed

    FunnyBird supplies the clean renderer reference that CUB70 lacks. The patch
    method passes only if:

    - all five FunnyBird parts are represented;
    - known grounded controls wing and foot are hurt more by target patches than by
      other-bird patches;
    - this adjusted effect grows with dose for both fills;
    - the two fills broadly agree on part ordering;
    - patch responses have a positive row-level relation with clean renderer
      deletion; and
    - at least four parts show a positive target probability drop under each fill.

    Failure stops the driver before CUB70 is run. Passing licenses only a test of
    **robust local pixel reliance**, not a renderer-quality causal swap.
    """, "gate"),

    code(r"""
    from pathlib import Path
    import json
    import os
    import pandas as pd
    from PIL import Image
    from IPython.display import display

    PATCH_ROOT = Path(os.environ["CURATED_DATA"]) / "randomized_patch_masking"
    PATCH_COMPARE = PATCH_ROOT / "comparison"
    PATCH_AUDIT = PATCH_COMPARE / "randomized_patch_audit.json"
    required = [PATCH_ROOT / "funnybirds-cbm-s1.parquet",
                PATCH_ROOT / "cub70-cbm-s1.parquet", PATCH_AUDIT,
                PATCH_COMPARE / "funnybird_patch_calibration.png",
                PATCH_COMPARE / "funnybirds_patch_dose_raw_z.png",
                PATCH_COMPARE / "funnybirds_patch_dose_summary.png",
                PATCH_COMPARE / "cub70_patch_dose_raw_z.png",
                PATCH_COMPARE / "cub70_patch_dose_summary.png"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run bash analysis/run_randomized_patch_masking.sh first; missing:\n" +
                                "\n".join(missing))
    patch_audit = json.loads(PATCH_AUDIT.read_text())
    assert patch_audit["funnybird_calibration"]["status"] == "PASS", patch_audit
    print("[RANDOMIZED PATCH INPUT PASS]")
    print(json.dumps(patch_audit["funnybird_calibration"], indent=2))
    """, "load",
         "Audit output proving that the FunnyBird randomized-patch calibration passed before CUB70 was evaluated."),

    md(r"""
    ### Figure P1 · FunnyBird calibration

    **Question.** Does the weak patch test recover known behavior from the clean
    renderer intervention?

    **How to read it.** The left panel compares clean whole-part probability drop
    with highest-dose patch drop. The right panel requires two things at once:
    target patches hurt more than other-bird patches, and that difference grows as
    dose rises. Marker shape identifies the fill method.

    Do not interpret CUB70 until this figure and every saved FunnyBird intervention
    sheet have been displayed and visually accepted.
    """, "calibration_header"),

    code(r"""
    display(Image.open(PATCH_COMPARE / "funnybird_patch_calibration.png"))
    display(pd.read_csv(PATCH_COMPARE / "funnybird_patch_calibration_by_part.csv").round(4))
    """, "calibration",
         "FunnyBird calibration comparing clean renderer deletion with small randomized patches, including dose slopes and both fill methods."),

    md(r"""
    ### Figure P2 · Inspect the masks and both fills

    Each sheet shows the original image followed by target, other-bird, and
    background masks at the largest requested dose. Each mask is then shown with
    local-blur and local-mean fill. The colored overlay must remain a collection of
    small soft patches rather than one large whole-part blob. Controls must not
    systematically land on a more important named part.
    """, "examples_header"),

    code(r"""
    for dataset_name in ["funnybirds-cbm-s1", "cub70-cbm-s1"]:
        root = PATCH_ROOT / f"{dataset_name}_examples"
        print(dataset_name)
        for path in sorted(root.glob("*.png")):
            print(path.name)
            display(Image.open(path))
    """, "examples",
         "All saved FunnyBird and CUB70 small-mask intervention sheets, showing target, other-bird, and background masks with both fills."),

    md(r"""
    ### Figure P3 · FunnyBird dose response in raw `z`

    **Question.** When more local evidence is hidden, does the matching raw concept
    score move down more than the controls?

    Each row is one part and each column is one fill. Red is target masking, blue is
    other-bird masking, and green is background masking. The y-axis is
    `delta_z` divided by that exact concept's ordinary `z` spread only to place
    different concept slots on one display. The original raw values remain saved.

    A reliable local effect is a red curve that falls with dose and lies below both
    controls under both fills. One isolated point is insufficient.
    """, "fb_raw_header"),

    code(r"""
    display(Image.open(PATCH_COMPARE / "funnybirds_patch_dose_raw_z.png"))
    display(pd.read_csv(PATCH_COMPARE / "funnybirds_patch_summary.csv").round(4))
    """, "fb_raw",
         "FunnyBird raw-concept-score dose responses for target, other-bird, and background patches under blur and local-mean fills."),

    md(r"""
    ### Figure P4 · CUB70 dose response in raw `z`

    This is the identical test and display, now permitted only because Gate 1 passed.
    We do not assume that CUB70 must reproduce FunnyBird's part ordering. We ask which
    mapped CUB70 parts show a dose-dependent target-specific response under both fills,
    and which do not.
    """, "cub_raw_header"),

    code(r"""
    display(Image.open(PATCH_COMPARE / "cub70_patch_dose_raw_z.png"))
    display(pd.read_csv(PATCH_COMPARE / "cub70_patch_summary.csv").round(4))
    """, "cub_raw",
         "CUB70 raw-concept-score dose responses for target, other-bird, and background patches under blur and local-mean fills."),

    md(r"""
    ### Figures P5–P6 · target-specific drop and retained probability

    The left panel asks whether the target is more important than other bird pixels:
    positive `adjusted_drop_p` supports target-specific local reliance. The right
    panel asks how much probability remains after partial masking. A high retained
    value does not cancel a positive local effect; it says that the tested patches
    mattered but did not exhaust the model's support.

    This is where the two hypotheses separate:

    - **red/raw target effect without retention:** the part pixels dominate;
    - **target effect plus high retention:** pixels matter and other evidence also
      sustains the answer;
    - **no target-specific dose effect:** either the model is locally insensitive or
      this weak mask test cannot identify the relevant pixels;
    - **fills disagree:** replacement artifact remains a live alternative.
    """, "summary_header"),

    code(r"""
    print("FunnyBird")
    display(Image.open(PATCH_COMPARE / "funnybirds_patch_dose_summary.png"))
    print("CUB70")
    display(Image.open(PATCH_COMPARE / "cub70_patch_dose_summary.png"))
    """, "summary",
         "FunnyBird and CUB70 probability-scale summaries of target-specific dose response and probability retained after partial masking."),

    md(r"""
    ### Acceptance rule and next question

    No interpretation is written in advance. After execution, display Figures P1–P6
    and every intervention sheet in chat. For each figure record: literal observation,
    strongest alternative explanation, discriminating test, limited conclusion, and
    next question.

    Even a clean pass establishes only repeated local pixel reliance and partial
    contextual retention. It cannot create a donor part, hold a real bird's pose and
    texture fixed, or prove that retained support came specifically from species.
    FunnyBird's renderer swap remains the stronger causal intervention. The CUB70
    exact-concept species residual can be combined with this result only as converging
    evidence, not as proof by itself.
    """, "decision"),
]


def main() -> None:
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb["cells"] = [cell for cell in nb["cells"]
                       if cell.get("metadata", {}).get("paired_section") != SECTION]
        nb["cells"].extend(json.loads(json.dumps(CELLS)))
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"inserted {len(CELLS)} randomized-patch cells into {path}")


if __name__ == "__main__":
    main()
