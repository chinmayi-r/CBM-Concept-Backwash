#!/usr/bin/env python3
"""Append the limited CUB70 beak/tail insertion pilot to notebook 05."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_cub_cbm.ipynb"
SECTION = "cub70_beak_tail_swap_pilot_v1"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(text).strip().splitlines()]


def md(text: str, name: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"cub05_section": SECTION},
            "source": lines(text), "id": f"cubpilot-{name}"}


def code(text: str, name: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"cub05_section": SECTION}, "source": lines(text),
            "id": f"cubpilot-{name}"}


CELLS = [
    md(r"""
    ## Final controlled approximation: visible beak/tail insertion pilot

    ### Why FunnyBird is proved but not fully explained

    FunnyBird's validated renderer gives a direct causal result: replacing one
    part moves the model toward the donor concept, yet the old source concept
    often remains stronger. That proves backwash exists.

    It does **not** mean every failure has been explained. Visibility-conflicting
    labels, exact variant difficulty, and source-species/body context each explain
    part of the pattern. Filtering or adjusting for them never makes the remaining
    failures disappear. The body/species explanation is also observational because
    FunnyBird body context was not independently changed while holding the part
    fixed. The correct conclusion is therefore: multiple demonstrated or supported
    contributors plus an unexplained residual.

    ### CUB question

    Can a crude but controlled CUB edit at least reproduce the central direction?

    For a clearly visible beak or tail with source value `S`, choose:

    - another visible example with the same value `S`;
    - a visible donor with a different value `D` from the same attribute family.

    Resize each donor part into the same target mask. Score the target original,
    target deletion, same-value paste, and different-value paste.

    **Primary variable**

    `donor response = (z_D - z_S) after different-value paste`
    `                 - (z_D - z_S) after same-value paste`

    In simple language: both images went through the same ugly copy-and-resize
    process. Does changing the pasted *value* specifically move the model toward
    that new value?

    **Prediction**

    - Positive response: the model noticed the donor pixels.
    - Positive response but final `z_D-z_S < 0`: the new pixels helped, but the old
      value/context still won. This is a candidate CUB analogue of backwash.
    - No response: this edit supplies no evidence for the mechanism.

    **Boundary**

    This cannot equal FunnyBird's renderer. Resizing, mask quality, donor pose,
    lighting, and donor-species appearance can still cause the score change. Every
    saved edit must therefore be shown and inspected before numerical interpretation.
    """, "question"),

    code(r"""
    from pathlib import Path
    import json, os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from IPython.display import display

    PILOT_ROOT = Path(os.environ["CURATED_DATA"]) / "cub70_beak_tail_swap_pilot"
    PILOT_DATA = PILOT_ROOT / "cub70-cbm-s1.parquet"
    PILOT_AUDIT = PILOT_DATA.with_suffix(".audit.json")
    PILOT_SUMMARY = PILOT_DATA.with_suffix(".summary.csv")
    PILOT_EXAMPLES = PILOT_ROOT / "cub70-cbm-s1_examples"
    PILOT_READY = all(p.exists() for p in [PILOT_DATA, PILOT_AUDIT, PILOT_SUMMARY])
    if not PILOT_READY:
        print("PENDING: run bash analysis/run_cub70_beak_tail_swap_pilot.sh inside an allocated GPU session")
    else:
        audit = json.loads(PILOT_AUDIT.read_text())
        if audit.get("status") != "COMPUTATION_PASS_VISUAL_REVIEW_REQUIRED":
            raise RuntimeError(f"unexpected pilot status: {audit.get('status')}")
        PILOT = pd.read_parquet(PILOT_DATA)
        display(pd.read_csv(PILOT_SUMMARY).round(4))
        print(json.dumps(audit, indent=2))
    """, "load"),

    md(r"""
    ### Figure C1 · Inspect every edit first

    Each sheet shows the target photograph and mask, deletion, same-value donor
    and paste, then different-value donor and paste. A numerical result is rejected
    if the pasted beak/tail is clearly misplaced, covers another region, or becomes
    an obvious rectangular/global-colour artifact.
    """, "examples_header"),

    code(r"""
    if PILOT_READY:
        sheets = sorted(PILOT_EXAMPLES.glob("*.png"))
        if not sheets:
            raise RuntimeError("pilot passed computation but saved no visual sheets")
        for path in sheets:
            print(path.name)
            display(Image.open(path))
    """, "examples"),

    md(r"""
    ### Figure C2 · Did the donor value move the correct raw-score margin?

    Each point is one edited photograph. Above zero means the different-value paste
    moved `z_D-z_S` farther toward the donor than the same-value paste did.

    This is the closest CUB approximation to the FunnyBird within-image swap
    response. It is not accepted until Figure C1 passes visual review.
    """, "response_header"),

    code(r"""
    if PILOT_READY:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        rng = np.random.default_rng(12)
        for i, (part, d) in enumerate(PILOT.groupby("part")):
            axes[0].scatter(np.full(len(d), i) + rng.uniform(-.12, .12, len(d)),
                            d.response_vs_source_control, alpha=.65, s=24,
                            label=part)
        axes[0].axhline(0, color="black", lw=1)
        parts = sorted(PILOT.part.unique())
        axes[0].set_xticks(range(len(parts))); axes[0].set_xticklabels(parts)
        axes[0].set_ylabel("donor response in raw z margin")
        axes[0].set_title("Different-value paste minus same-value paste")

        for part, d in PILOT.groupby("part"):
            axes[1].scatter(d.response_vs_source_control, d.margin_donor_insert,
                            alpha=.65, s=26, label=part)
        axes[1].axvline(0, color="black", lw=1)
        axes[1].axhline(0, color="black", lw=1)
        axes[1].set_xlabel("donor response")
        axes[1].set_ylabel("final donor minus source z")
        axes[1].set_title("Noticed donor pixels, and did the donor finally win?")
        axes[1].legend()
        plt.tight_layout(); plt.show()

        literal = PILOT.groupby("part").agg(
            pairs=("image", "size"),
            median_response=("response_vs_source_control", "median"),
            response_positive=("response_vs_source_control", lambda x: (x > 0).mean()),
            median_final_margin=("margin_donor_insert", "median"),
            donor_wins=("donor_wins_after_insert", "mean"),
            candidate_retained_source=("candidate_retained_source", "mean"),
        )
        display(literal.round(4))
    """, "response"),

    md(r"""
    ### Executed result · the pilot does not pass its scientific gate

    **Literal observation.** Forty beak and forty tail pairs passed the mechanical
    selection checks. The median different-value-minus-same-value donor response
    was `-0.0037` for beak and `0.0000` for tail. Only 40% of pairs moved in the
    predicted positive direction for either part. The final donor-minus-source
    margin remained negative at the median (`-2.54` beak; `-1.51` tail), but that
    cannot be called retained-source backwash because the required positive donor
    response did not occur first. Deleting the named part was also inconsistent:
    the source score fell in only 47.5% of cases for each part.

    **Visual observation.** All sixteen saved sheets were inspected in chat. Most
    masks are on the intended beak or tail, but beak changes are often only a few
    pixels. Several tail pastes look like flat colour/texture strips, and the
    hummingbird beak edit is visibly distorted. The numerical tail mean is also
    pulled by a few large responses while its median remains zero.

    **Alternative explanations.** A real CUB donor effect may be too small for this
    resize-and-paste operation; the selected exact attributes may not be expressed
    by the pasted pixels alone; or resizing, masks, pose, lighting, and donor-species
    texture may obscure it. The present data cannot distinguish those possibilities.

    **Limited conclusion.** This pilot supplies no reliable within-image CUB donor
    response and therefore does not prove CUB backwash. It does not weaken the clean
    FunnyBird renderer result. CUB currently supports the observational ingredients
    — visibility-conflicting labels, uneven visibility, heterogeneous concept
    difficulty, and species information in concept predictions — but not the same
    causal mechanism at FunnyBird strength.

    **Next question.** Stop tuning this pilot. Preserve the failure, finish notebook
    05 with this boundary, and let notebook 06 ask only the narrower CUB minimality
    question: how compression changes the already-observed CUB concept behavior.
    """, "executed_result"),

    md(r"""
    ### Interpretation rule and stopping point

    **Literal observation first.** Report the sign and spread of the donor response,
    then the final donor-minus-source margin. Do not describe a median without also
    checking the individual points and sheets.

    **Alternative explanations.** A positive response can come from the named visual
    value, donor-species texture, colour/lighting mismatch, resizing, or mask shape.
    A negative final margin can come from source/context retention, a poor paste, or
    an intrinsically weak donor concept.

    **Limited conclusion.** If the visual sheets are plausible and responses are
    broadly positive, CUB shows within-image sensitivity in the same direction as
    FunnyBird. If many final margins remain negative, call that a *candidate residual
    source/context effect*, not full causal proof of CUB backwash. If sheets or response
    direction fail, retain the failure and stop; do not tune until the desired result
    appears.

    **Next question.** Only a successful pilot justifies a larger, preregistered CUB
    insertion run. MCBM comes after this CBM decision and must not rewrite it.
    """, "decision"),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [c for c in nb["cells"]
                   if c.get("metadata", {}).get("cub05_section") != SECTION]
    nb["cells"].extend(CELLS)
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"inserted {len(CELLS)} pilot cells into {NOTEBOOK}")


if __name__ == "__main__":
    main()
