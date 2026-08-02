#!/usr/bin/env python3
"""Append the same cross-dataset deletion section to notebooks 02 and 05.

The section is deliberately reciprocal: neither notebook receives a weaker
summary of the other dataset.  Both display the same FunnyBird/CUB70 inputs,
figures, component scores, caveats, and decision rules.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

CURATED = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    CURATED / "notebooks" / "02_funnybirds_cbm.ipynb",
    CURATED / "notebooks" / "05_cub_cbm.ipynb",
]
SECTION = "fb_cub70_paired_deletion_v1"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def cell_id(name: str) -> str:
    return "pair-" + hashlib.sha1(name.encode()).hexdigest()[:12]


def md(text: str, name: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id(name),
            "metadata": {"paired_section": SECTION, "paired_cell": name},
            "source": lines(text)}


def code(text: str, name: str, alt: str | None = None) -> dict:
    metadata = {"paired_section": SECTION, "paired_cell": name}
    if alt:
        metadata["alt"] = alt
    return {"cell_type": "code", "id": cell_id(name), "execution_count": None,
            "outputs": [], "metadata": metadata, "source": lines(text)}


CELLS = [
    md(r"""
    ## Direct competition: the same part-deletion test on FunnyBirds and CUB70

    We now stop giving the two datasets different tests. For every eligible
    image and every **exact positive concept** `j`, we score the same four images:

    1. `z_original`: the ordinary image;
    2. `z_target_deleted`: its named part is removed;
    3. `z_control_deleted`: the same-shaped patch is removed elsewhere on the bird;
    4. `z_part_only`: the named part remains sharp while the rest is blurred.

    The main number is

    `target_minus_control_z = z_target_deleted - z_control_deleted`.

    In simple language: **after equal-sized damage, is the exact concept score lower
    when we remove its own part than when we damage another place?** A negative value
    says yes. We always print all four scores too, because the difference alone can hide
    whether both scores rose, both fell, or only one moved.

    This test has the same code and raw-logit (`z_j`) definition on both datasets.
    It is not the clean FunnyBird renderer deletion: image inpainting can leave artifacts.
    FunnyBird therefore calibrates whether this weaker, shared intervention agrees with
    the already accepted renderer-based result.
    """, "header"),

    md(r"""
    ### One evidence matrix for both datasets

    | Question | FunnyBird | CUB70 | What counts as a direct comparison? |
    |---|---|---|---|
    | Is the concept predictable before intervention? | existing model-health plots | existing model-health and collapse plots | exact-concept raw `z`, variation, and task accuracy |
    | Is the named part actually visible? | exact renderer part map | released CUB70 masks | visible target pixels on the tested image |
    | Are labels positive when the part is absent? | existing visibility/label-conflict analysis | existing zero/tiny/visible-mask analysis | same positive-label conflict definition |
    | Can species be decoded from concepts? | existing held-out species probe | existing matched species probe | held-out species prediction from concept outputs |
    | Does removing the named part matter? | clean renderer deletion **and** shared mask deletion | shared mask deletion | exact `z_j`, target deletion versus same-shape control |
    | Does context retain the concept without the part? | shared target-deleted and part-only inputs | same shared inputs | all four component scores, not only a difference |
    | Does source species organize the residual? | exact-concept-conditioned residual | same calculation | species eta-squared after fixing exact concept |
    | Does inserting a donor part change ordering? | clean renderer swap exists | no clean renderer | CUB copy-paste stays quarantined until FunnyBird calibration passes |
    | Does minimality or relabeling change it? | notebooks 03 and 03rl | notebook 06/pending follow-up | only after this non-RL CBM baseline is accepted |

    The rows are not all equally causal. The shared deletion is the next test because it
    is the strongest operation that can be applied to both datasets now. The swap row
    remains secondary because CUB70 has no renderer that can change a bird part while
    holding everything else fixed.
    """, "matrix"),

    md(r"""
    ### Question and prediction

    **Question.** Does the model use the visible pixels of the named part, and does it
    still keep the concept active when those pixels are removed?

    **Backwash prediction.** If species/body context is filling in a missing part, then:

    - target deletion should hurt more than control deletion;
    - but `z_target_deleted` can remain above zero;
    - and the deleted-part score can vary systematically with source species even after
      we hold the exact concept fixed.

    None of these alone proves species backwash. The claim becomes stronger only when
    mask quality, model collapse, deletion artifacts, pose/scale, and exact-concept
    imbalance have been checked, and when the pattern repeats across concepts, species,
    images, and seeds.
    """, "question"),

    code(r"""
    from pathlib import Path
    import json
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from IPython.display import display

    PAIR_ROOT = Path(os.environ["CURATED_DATA"]) / "paired_deletion"
    FB_PAIR = PAIR_ROOT / "funnybirds-cbm-s1.parquet"
    CUB_PAIR = PAIR_ROOT / "cub70-cbm-s1.parquet"
    PAIR_COMPARE = PAIR_ROOT / "comparison"
    required = [FB_PAIR, CUB_PAIR, FB_PAIR.with_suffix(".audit.json"),
                CUB_PAIR.with_suffix(".audit.json"),
                PAIR_COMPARE / "paired_deletion_audit.json",
                PAIR_COMPARE / "funnybird_deletion_calibration.png",
                PAIR_COMPARE / "paired_deletion_main.png",
                PAIR_COMPARE / "paired_deletion_species.png"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Run bash analysis/run_paired_deletion.sh first; missing:\n" +
                                "\n".join(missing))
    pair_audit = json.loads((PAIR_COMPARE / "paired_deletion_audit.json").read_text())
    assert pair_audit["status"] == "PASS", pair_audit
    fb_input_audit = json.loads(FB_PAIR.with_suffix(".audit.json").read_text())
    cub_input_audit = json.loads(CUB_PAIR.with_suffix(".audit.json").read_text())
    FB_PAIR_DF = pd.read_parquet(FB_PAIR)
    CUB_PAIR_DF = pd.read_parquet(CUB_PAIR)
    print("[SHARED INPUT PASS]", pair_audit)
    print("FunnyBird selection:", fb_input_audit["selection_counts"])
    print("CUB70 selection:", cub_input_audit["selection_counts"])
    """, "load"),

    md(r"""
    ### Figure 0 · Calibration: does this shared deletion behave like the clean one?

    Before using mask inpainting on CUB70, we run it on FunnyBird, where clean
    renderer deletion already exists. Each point is one part. Both axes are the drop in
    the same exact concept's probability. Agreement in sign and part ordering means the
    weaker shared test is at least tracking the known intervention; disagreement puts
    the CUB70 deletion result in quarantine.
    """, "calibration_header"),

    code(r"""
    display(Image.open(PAIR_COMPARE / "funnybird_deletion_calibration.png"))
    display(pd.read_csv(PAIR_COMPARE / "funnybird_deletion_calibration.csv").round(3))
    print("CALIBRATION:", pair_audit["funnybird_calibration"])
    """, "calibration",
         "Per-part comparison of clean FunnyBird renderer deletion and the shared mask-inpainting deletion used for both datasets."),

    md(r"""
    ### Figure A · Inspect the actual interventions before accepting the numbers

    Every row uses the same layout: original, target mask, target deletion, equal-shape
    control mask, control deletion, and part only. FunnyBird is on the left and CUB70 is
    on the right. We must inspect every mapped part before interpreting the score plots.
    """, "examples_header"),

    code(r"""
    common_parts = ["tail", "wing", "beak", "leg", "eye"]
    example_roots = {
        "FunnyBird": FB_PAIR.with_suffix("").parent / (FB_PAIR.stem + "_examples"),
        "CUB70": CUB_PAIR.with_suffix("").parent / (CUB_PAIR.stem + "_examples"),
    }
    aliases = {"FunnyBird": {"leg": "foot"}, "CUB70": {}}
    fig, axes = plt.subplots(len(common_parts), 2, figsize=(18, 3.2 * len(common_parts)))
    for row, part in enumerate(common_parts):
        for col, dataset_name in enumerate(["FunnyBird", "CUB70"]):
            local_part = aliases[dataset_name].get(part, part)
            files = sorted(example_roots[dataset_name].glob(f"{local_part}_*.png"))
            axes[row, col].axis("off")
            if files:
                axes[row, col].imshow(Image.open(files[0]))
                axes[row, col].set_title(f"{dataset_name} · {part}")
            else:
                axes[row, col].text(.5, .5, f"no eligible {part} example",
                                    ha="center", va="center")
    plt.suptitle("Same intervention examples: FunnyBird and CUB70", y=1.002)
    plt.tight_layout(); plt.show()
    """, "examples",
         "Paired FunnyBird and CUB70 intervention sheets for tail, wing, beak, leg, and eye, each showing original, masks, deletions, and part-only input."),

    md(r"""
    ### Figure B · Literal shared deletion result

    The left panel shows the exact-concept target-minus-control score in raw-z units,
    divided only by that concept's ordinary-score spread so the two separately trained
    models can share an axis. Negative means deleting the named part hurt more.

    The right panel uses only yes/no outcomes: target hurt more; score stayed positive
    after target deletion; deleted-context score beat the part-only score.
    """, "main_header"),

    code(r"""
    display(Image.open(PAIR_COMPARE / "paired_deletion_main.png"))
    display(pd.read_csv(PAIR_COMPARE / "paired_deletion_summary.csv").round(3))
    """, "main_figure",
         "FunnyBird and CUB70 comparison of standardized exact-concept target-versus-control deletion and three scale-free paired outcomes."),

    md(r"""
    ### Component check · do not let subtraction hide the mechanism

    This table prints the median original, target-deleted, control-deleted, and part-only
    raw `z_j` separately. It answers whether a negative adjusted deletion came from the
    expected target drop rather than a strange control rise.
    """, "components_header"),

    code(r"""
    pieces = []
    for label, frame in [("FunnyBird", FB_PAIR_DF), ("CUB70", CUB_PAIR_DF)]:
        d = frame.copy()
        d["part_common"] = d.part.replace({"foot": "leg"})
        block = (d.groupby("part_common")[["z_original", "z_target_deleted",
                                           "z_control_deleted", "z_part_only"]]
                 .median().reset_index())
        block.insert(0, "dataset", label)
        pieces.append(block)
    display(pd.concat(pieces, ignore_index=True).round(3))
    """, "components"),

    md(r"""
    ### Figure C · does source species still organize the deleted-part score?

    For each exact concept separately, this plot asks how much of the remaining
    `z_target_deleted` variation lines up with source species. The effect is corrected
    for the number of species groups and checked against 200 random label shuffles.
    This is the species
    backwash candidate. It is **observational**, not causal: species also carries pose,
    body appearance, and background. The target/control and part-only tests above must
    carry the causal burden; this plot only identifies the suspected source of the
    remaining prediction.
    """, "species_header"),

    code(r"""
    display(Image.open(PAIR_COMPARE / "paired_deletion_species.png"))
    display(pd.read_csv(PAIR_COMPARE / "paired_deletion_species_residual.csv")
            .sort_values(["dataset", "part_common", "species_omega2"],
                         ascending=[True, True, False]).head(40).round(3))
    """, "species_figure",
         "FunnyBird and CUB70 bias-corrected source-species effects after fixing the exact concept and deleting its mapped part."),

    md(r"""
    ### Alternatives, decision rule, and next question

    We may say **part pixels matter** only if target deletion is visibly valid and hurts
    more than the same-shaped control. We may say **context remains sufficient for some
    predictions** only if the target-deleted score stays positive and exceeds part-only.
    We may call **species backwash a supported explanation** only if the species residual
    survives exact-concept conditioning and repeats broadly, while model-collapse and
    mask-quality exclusions do not explain it.

    A crude copy-paste swap is deliberately secondary. It will be interpreted on CUB70
    only after the identical copy-paste operation on FunnyBird agrees in direction with
    FunnyBird's clean renderer swap. Until that calibration passes, it is an image-editing
    artifact check—not evidence.

    **Next question after inspecting Figures A–C:** which specific part/concept/species
    rows violate the prediction, and are those failures explained by poor masks, pose or
    scale, collapsed outputs, or genuine context-driven scores?
    """, "decision"),
]


def main() -> None:
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        nb["cells"] = [c for c in nb["cells"]
                       if c.get("metadata", {}).get("paired_section") != SECTION]
        # Copy the cells so modifications during later execution cannot leak between notebooks.
        nb["cells"].extend(json.loads(json.dumps(CELLS)))
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"inserted {len(CELLS)} reciprocal paired-deletion cells into {path}")


if __name__ == "__main__":
    main()
