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

    A blur over an almost uniform region can sometimes change no RGB value. Such a
    row is not model evidence because no intervention occurred. If either the target
    or other-bird edit is a no-op, the complete matched unit—including its background
    control—is dropped. The audit prints these losses and fails if any part/fill or
    the dose-response coverage disappears.

    Failure stops the driver before CUB70 is run. Passing licenses only a test of
    **robust local pixel reliance**, not a renderer-quality causal swap.

    **Executed outcome (2026-08-02): FAIL.** The computation completed, but the
    calibration did not pass. Only 166 FunnyBird images survived the complete
    matching rules, and the surviving parts were beak, eye, foot, and tail.
    **Wing was absent.** Therefore this run cannot compare wing with tail and cannot
    license the CUB70 stage. The failure is preserved below rather than hidden.
    """, "gate"),

    code(r"""
    from pathlib import Path
    import json
    import os
    import pandas as pd
    from PIL import Image
    from IPython.display import display

    PATCH_ROOT = Path(os.environ["CURATED_DATA"]) / "randomized_patch_masking"
    PATCH_CALIBRATION = PATCH_ROOT / "calibration"
    PATCH_FULL_COMPARE = PATCH_ROOT / "comparison"
    PATCH_AUDIT = PATCH_CALIBRATION / "randomized_patch_audit.json"
    PATCH_INPUT_AUDIT = PATCH_ROOT / "funnybirds-cbm-s1.audit.json"
    required = [PATCH_ROOT / "funnybirds-cbm-s1.parquet", PATCH_AUDIT,
                PATCH_CALIBRATION / "funnybird_patch_calibration.png",
                PATCH_CALIBRATION / "funnybirds_patch_dose_raw_z.png",
                PATCH_CALIBRATION / "funnybirds_patch_dose_summary.png"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run bash analysis/run_randomized_patch_masking.sh first; missing:\n" +
                                "\n".join(missing))
    patch_audit = json.loads(PATCH_AUDIT.read_text())
    PATCH_CALIBRATION_PASSED = patch_audit["funnybird_calibration"]["status"] == "PASS"
    PATCH_CUB_READY = (PATCH_CALIBRATION_PASSED and
                       (PATCH_ROOT / "cub70-cbm-s1.parquet").exists() and
                       (PATCH_FULL_COMPARE / "randomized_patch_audit.json").exists())
    PATCH_COMPARE = PATCH_FULL_COMPARE if PATCH_CUB_READY else PATCH_CALIBRATION
    print("[RANDOMIZED PATCH COMPUTATION LOADED]")
    print("FUNNYBIRD CALIBRATION:", "PASS" if PATCH_CALIBRATION_PASSED else "FAIL")
    print("CUB70:", "AVAILABLE" if PATCH_CUB_READY else "NOT RUN / NOT ADMISSIBLE")
    print(json.dumps(patch_audit["funnybird_calibration"], indent=2))
    if PATCH_INPUT_AUDIT.exists():
        patch_input_audit = json.loads(PATCH_INPUT_AUDIT.read_text())
        print("FUNNYBIRD INPUT / SELECTION AUDIT")
        print(json.dumps({
            "images": patch_input_audit.get("images"),
            "parts": patch_input_audit.get("parts"),
            "selection_counts": patch_input_audit.get("selection_counts"),
            "no_op_rows_by_fill_and_location": patch_input_audit.get(
                "no_op_rows_by_fill_and_location"),
            "post_gate_coverage": patch_input_audit.get("post_gate_coverage"),
        }, indent=2))
    else:
        print("Selection audit not found; do not guess why wing was excluded.")
    """, "load",
         "Audit output recording the FunnyBird calibration result and the exact selection and no-op losses."),

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
    **Literal result.** The clean renderer removes a whole part and drops its
    concept probability by about 1. The small patches remove only part of a part.
    On the probability scale, nearly every patch result remains at zero drop;
    local-mean beak is the only clearly separated point (about 0.018). Wing is
    missing entirely.

    **Why this failed.** The preregistered check required all five parts and a
    positive wing/foot control. Those requirements were not met. Also, the CBM's
    original probabilities are often extremely close to 1. For example, a raw
    score can fall from `z=20` to `z=16` while both sigmoid probabilities still
    print as `1.000`. Probability therefore hides real but partial score movement.

    **Limited conclusion.** This figure does not say that tail is better grounded
    than wing. It says this partial-mask calibration cannot reproduce the complete
    clean-deletion ordering and cannot be transferred to CUB70.
    """, "calibration_result"),

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
        paths = sorted(root.glob("*.png"))
        print(dataset_name, f"({len(paths)} saved sheets)")
        if not paths:
            print("No sheets: dataset was not run or no example survived.")
        for path in paths:
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
    **Literal result.** For every surviving part, the red target curve moves down
    as more of that part is covered, while the blue other-bird and green background
    controls remain near zero. The effect is strongest for beak. At the largest
    dose it is smaller for eye and foot, and smallest for tail. Local-mean fill
    causes a larger fall than blur, but both fills give the same broad direction.
    Each panel has its own y-axis, so compare the printed values, not just line
    steepness.

    **What the controls rule against.** If any edit anywhere caused the score to
    fall, blue and green would fall with red. They mostly do not. If only one fill
    texture caused the result, blur and local mean would disagree in direction.
    They do not. This supports real local-pixel use in the selected beak, eye,
    foot, and tail examples.

    **What the controls do not rule out.** The sample is heavily selected, wing is
    absent, a translated control can still land near another useful part, and a
    partial patch is not a donor-part swap. Thus the result cannot rank all five
    parts or explain the remaining tail score as species backwash.
    """, "fb_raw_result"),

    md(r"""
    ### Figure P4 · CUB70 dose response in raw `z`

    This is the identical test and display, now permitted only because Gate 1 passed.
    We do not assume that CUB70 must reproduce FunnyBird's part ordering. We ask which
    mapped CUB70 parts show a dose-dependent target-specific response under both fills,
    and which do not.
    """, "cub_raw_header"),

    code(r"""
    if PATCH_CUB_READY:
        display(Image.open(PATCH_COMPARE / "cub70_patch_dose_raw_z.png"))
        display(pd.read_csv(PATCH_COMPARE / "cub70_patch_summary.csv").round(4))
    else:
        print("CUB70 was correctly not run because the FunnyBird calibration failed.")
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
    if PATCH_CUB_READY:
        print("CUB70")
        display(Image.open(PATCH_COMPARE / "cub70_patch_dose_summary.png"))
    else:
        print("CUB70 summary withheld: FunnyBird calibration did not pass.")
    """, "summary",
         "FunnyBird and CUB70 probability-scale summaries of target-specific dose response and probability retained after partial masking."),

    md(r"""
    ### V2 follow-up: all five parts were recovered, but one control was unfair

    V1 could not place the wide wing control. V2 moved each small control patch
    separately and switched the main readout from saturated probability to raw
    concept score `z`. We predicted that all five parts would now appear and that
    blur and mean fill would rank the parts similarly.

    **Result:** five of six checks passed. The only failure was agreement between
    fills on part ranking. CUB70 therefore still did not run.
    """, "v2_header"),

    code(r"""
    PATCH_V2 = Path(os.environ["CURATED_DATA"]) / "randomized_patch_masking_v2"
    PATCH_V2_CAL = PATCH_V2 / "calibration"
    PATCH_V2_AUDIT = json.loads(
        (PATCH_V2_CAL / "randomized_patch_audit.json").read_text())
    PATCH_V2_INPUT = json.loads(
        (PATCH_V2 / "funnybirds-cbm-s1.audit.json").read_text())
    print(json.dumps(PATCH_V2_AUDIT["funnybird_calibration"], indent=2))
    display(pd.read_csv(
        PATCH_V2_CAL / "funnybird_patch_calibration_by_part.csv").round(4))
    for figure in ["funnybird_patch_calibration.png",
                   "funnybirds_patch_dose_raw_z.png",
                   "funnybirds_patch_dose_summary.png"]:
        print(figure)
        display(Image.open(PATCH_V2_CAL / figure))
    """, "v2_summary",
         "FunnyBird v2 raw-score calibration and dose-response figures for all five parts."),

    code(r"""
    v2_examples = PATCH_V2 / "funnybirds-cbm-s1_examples"
    for path in sorted(v2_examples.glob("*.png")):
        print(path.name)
        display(Image.open(path))
    """, "v2_examples",
         "Every saved FunnyBird v2 intervention sheet, including two wing examples and both fill methods."),

    md(r"""
    **Literal observation.** Target masking lowers the matching raw score for every
    part under both fills. Tail has the smallest target response. Wing has a large
    target response under both fills. The disagreement appears only after
    subtracting the wing other-bird control under mean fill: that control also
    lowers the wing score strongly.

    **Why that control is suspect.** The wing sheets show control patches spread
    over several meaningful non-wing regions. The implementation then fills all
    those separated patches with one median colour computed from one large box
    around them. That is not a separate local mean for each patch. It can damage
    the control more than the target and artificially shrink the adjusted wing
    effect.

    **Coverage warning.** V2 selected 100 wing examples, but 368 could not support
    the requested matched non-wing control. Only 11 wing images contributed to the
    final calibration. The wing direction is informative, but its size is not yet
    representative.

    **Limited conclusion.** The selected examples show real local pixel use: tail
    is weak and wing is strong. This still is not a direct backwash measure. Beak
    is the useful counterexample: masking beak pixels strongly changes `z_beak`,
    yet a donor swap can still fail if the old source/context score remains even
    stronger. Local response and source retention must be shown separately.

    **Next discriminating test.** Before another GPU run, use a mask-only preflight
    to choose a lower common dose schedule with adequate coverage for all five
    parts, and replace the global median colour with a genuinely local per-pixel or
    per-patch mean. Then run one final FunnyBird-only calibration. CUB70 remains
    blocked unless that calibration passes.
    """, "v2_result"),

    md(r"""
    ### Acceptance rule and next question

    **Executed decision.** This is a documented failed discriminating test, not
    positive CUB evidence. Keep Figures P1-P3 and every available FunnyBird sheet;
    Figures P4-P6 must say that CUB70 was withheld.

    The figures below record the literal observation, strongest alternative,
    limited conclusion, and next question after execution.

    Even a clean pass establishes only repeated local pixel reliance and partial
    contextual retention. It cannot create a donor part, hold a real bird's pose and
    texture fixed, or prove that retained support came specifically from species.
    FunnyBird's renderer swap remains the stronger causal intervention. The CUB70
    exact-concept species residual can be combined with this result only as converging
    evidence, not as proof by itself.

    **Why wing disappeared.** The input audit selected 100 wing examples normally,
    but wing has no pre-gate rows and no post-gate coverage. Therefore every wing
    repeat failed while constructing its controls, before model inference. The old
    method tried to translate the complete wide wing patch pattern as one rigid
    shape onto non-wing bird/background support. It could not place that shape.
    The later 65 no-op losses were all local-mean other-bird edits and did not cause
    the missing wing.

    **Next question.** A post-hoc v2 keeps the original failed run untouched, moves
    each small Gaussian control patch independently while matching patch count,
    softness, and total alpha mass, and uses standardized raw `z` as its primary
    calibration response. Probability remains a secondary saturation display.
    FunnyBird must still pass with all five parts before CUB70 runs.
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
