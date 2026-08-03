#!/usr/bin/env python3
"""Keep a plain-language account of the failed CUB whole-part deletion test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

PATH = Path(__file__).resolve().parents[1] / "notebooks" / "05_cub_cbm.ipynb"
SECTION = "cub_failed_whole_part_plain_v1"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def make_id(name: str) -> str:
    return "cubfail-" + hashlib.sha1(name.encode()).hexdigest()[:12]


def md(text: str, name: str) -> dict:
    return {"cell_type": "markdown", "id": make_id(name),
            "metadata": {"paired_section": SECTION, "paired_cell": name},
            "source": lines(text)}


def code(text: str, name: str, alt: str) -> dict:
    return {"cell_type": "code", "id": make_id(name), "execution_count": None,
            "outputs": [], "metadata": {"paired_section": SECTION,
                                           "paired_cell": name, "alt": alt},
            "source": lines(text)}


CELLS = [
    md(r"""
    ## What the previous CUB deletion test tried—and why it did not answer us

    Here is the experiment in ordinary language.

    For a photograph labelled, for example, **yellow beak**, we made four versions:

    1. the untouched photograph;
    2. the photograph with the beak mask filled in;
    3. the photograph with the same-shaped mask moved somewhere else on the bird;
    4. an extreme image containing mainly the beak.

    We then asked whether the model's yellow-beak score fell more in image 2 than
    image 3. The idea was reasonable: if the beak pixels supply the yellow-beak
    answer, covering the beak should hurt more than equally large damage elsewhere.

    The images below show why this particular implementation could not give a clean
    answer.
    """, "header"),

    code(r"""
    from pathlib import Path
    import os
    from PIL import Image
    from IPython.display import display

    # Do not depend on the later detailed paired-deletion section for paths.
    PAIR_ROOT = Path(os.environ["CURATED_DATA"]) / "paired_deletion"
    FB_PAIR = PAIR_ROOT / "funnybirds-cbm-s1.parquet"
    CUB_PAIR = PAIR_ROOT / "cub70-cbm-s1.parquet"

    simple_cases = []
    for label, root, pattern in [
        ("CUB70 beak example", CUB_PAIR.with_suffix("").parent /
         (CUB_PAIR.stem + "_examples"), "beak_*.png"),
        ("FunnyBird eye calibration example", FB_PAIR.with_suffix("").parent /
         (FB_PAIR.stem + "_examples"), "eye_*.png"),
        ("FunnyBird tail calibration example", FB_PAIR.with_suffix("").parent /
         (FB_PAIR.stem + "_examples"), "tail_*.png"),
    ]:
        matches = sorted(root.glob(pattern))
        if matches:
            simple_cases.append((label, matches[0]))
    if not simple_cases:
        raise FileNotFoundError("No saved whole-part deletion examples were found")
    for label, path in simple_cases:
        print(label)
        display(Image.open(path))
    """, "examples",
         "Concrete CUB70 beak and FunnyBird eye and tail examples from the failed whole-part mask-inpainting experiment."),

    md(r"""
    ### What is visibly wrong in these examples?

    **Example 1: a CUB70 beak.** Covering a released beak mask can create a large,
    smooth grey or colour-smeared blob. That blob is not a natural bird without a
    beak. When the same mask is moved for the control, it can cover the wing, chest,
    or another useful body region. We are then comparing “damage the beak” with
    “damage another real part,” not with harmless equal damage.

    **Example 2: the FunnyBird eye.** The clean FunnyBird renderer truly removes the
    eye and makes the eye-concept probability fall by almost one. Our filled-mask
    version produced almost no eye-score drop. Therefore the mask method failed to
    reproduce a result we already knew from a cleaner intervention.

    **Example 3: the FunnyBird tail.** The same disagreement happened for tail:
    clean renderer removal caused a large probability drop, while filled-mask
    deletion caused almost none. The model may have used context, but the fill may
    also have reconstructed enough tail colour or shape for the model. These two
    explanations look identical in the score.

    **Missing wing check.** The method could not find a valid same-shaped control for
    any FunnyBird wing example. Overall it kept only 148 of 2,500 possible FunnyBird
    image-parts. A test that drops almost everything—and drops an entire important
    control part—does not represent the original population well.
    """, "visible_problems"),

    md(r"""
    ### What was the consequence?

    | Result from that run | Tempting explanation | Why we cannot accept it |
    |---|---|---|
    | A concept score stayed high after its part was filled | “The model read the species instead of the part.” | The fill may have preserved or reconstructed useful colour and shape. |
    | A concept score fell strongly | “The named part supplied the answer.” | The large unnatural blob itself may have disturbed the model. |
    | Target and moved-control scores differed | “The target was uniquely important.” | The moved control often damaged another meaningful bird part. |
    | CUB70 scores still differed by species after fixing the exact concept | “This proves species backwash.” | Species also changes pose, body, background, and other attributes. This is an association, not a controlled cause. |

    Therefore the CUB70 whole-part deletion numbers are **quarantined**: we keep
    them as a documented failed test, but they cannot prove or disprove CUB70
    backwash. They do not damage the clean FunnyBird renderer evidence.

    What remains useful is the observational clue that many CUB70 exact-concept
    scores still vary by species. The next experiment tests whether small local
    patches produce a repeatable dose response without making one giant artificial
    hole. It uses two fill types and must pass on FunnyBird before CUB70 is run.
    """, "consequence"),
]


def main() -> None:
    nb = json.loads(PATH.read_text(encoding="utf-8"))
    cells = [cell for cell in nb["cells"]
             if cell.get("metadata", {}).get("paired_section") != SECTION]
    insert_at = None
    for index, cell in enumerate(cells):
        meta = cell.get("metadata", {})
        if (meta.get("paired_section") == "fb_cub70_randomized_patch_v1" and
                meta.get("paired_cell") == "header"):
            insert_at = index
            break
    if insert_at is None:
        raise RuntimeError("randomized-patch section not found; refusing ambiguous insertion")
    cells[insert_at:insert_at] = json.loads(json.dumps(CELLS))
    nb["cells"] = cells
    PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"inserted {len(CELLS)} plain failed-test cells into {PATH}")


if __name__ == "__main__":
    main()
