#!/usr/bin/env python3
"""Add the missing FunnyBird/CUB bridge and CUB70 example audit to notebook 05.

The inserted cells are deliberately executable from saved prediction parquets.  They
do not train a model and do not use Slurm.  Re-running this script replaces its own
cells without disturbing existing executed outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "05_cub_cbm.ipynb"
SECTION_ID = "cub05_comparative_audit"


def lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def cell(kind: str, source: str, suffix: str) -> dict:
    out = {
        "cell_type": kind,
        "metadata": {"cub05_section": SECTION_ID, "cub05_cell": suffix},
        "source": lines(source),
    }
    if kind == "code":
        out.update({"execution_count": None, "outputs": []})
    return out


CELLS = [
    cell("markdown", r"""
    ## 13 · Direct FunnyBird comparison and unresolved CUB examples

    This section keeps unlike measurements separate.

    ### Model variables

    In both datasets the encoder produces one raw score `z_j` for exact concept `j`.
    `prob_j` (called `c_preds` in FunnyBird files) is the sigmoid probability derived
    from that score. The species head reads the complete concept bottleneck.

    FunnyBirds has a controlled source image and donor part, so it can define:

    - `z_source,original`: old/source concept score before editing;
    - `z_donor,original`: donor concept score before editing, while donor part is absent;
    - `z_source,swap`: old/source score after its pixels were replaced;
    - `z_donor,swap`: inserted donor score after replacement;
    - `margin_swap = z_donor,swap - z_source,swap`;
    - `response_delta = margin_swap - (z_donor,original-z_source,original)`.

    CUB has no donor and no controlled swap. Its closest observational quantity is
    `visibility_effect = mean(prob_j | label_j=1, mask visible) -
    mean(prob_j | label_j=1, mask absent)`. It is **not** a swap margin and cannot by
    itself prove backwash.

    ### What can actually be paired

    | FunnyBird test | Closest CUB test | Same question? |
    |---|---|---|
    | ordinary task/concept accuracy | Figure 6 | yes, model-health guard |
    | species decoded from each part block | new CUB probe below | yes: stored species information; feature dimensions differ |
    | deletion retention | Figures 7/10/12 | only approximate; CUB has natural absence, not deletion |
    | donor-over-source swap ordering/margin | none | no CUB causal equivalent |
    | swap response delta | none | no donor/source pair in CUB |
    | swap success versus inserted pixel count | Figures 9/10 | approximate visibility-dose test |
    | variant confusion after a swap | Figures 7/12 plus exact-concept table | approximate; no CUB donor intervention |
    | controlled source-species residuals | Figure 11 | approximate; CUB still changes pose/background |
    | RLv2 labels changed by part | Figure 5 below as rates | related data audit, but CUB masks are test-only and no CUB training labels were changed |

    FunnyBird species decoding was `0.986` from raw `z` and `0.987` from concept
    probabilities, against `0.020` chance. Single-part accuracies were tail `0.193`,
    wing `0.119`, beak `0.082`, foot `0.081`, and eye `0.061`. These are close to
    `n_variants/50`: `0.18, 0.12, 0.08, 0.08, 0.06`. Thus the part block really stores
    the species partition associated with its canonical variant. The swap is still
    needed to establish that unchanged context can overpower newly inserted pixels.
    """, "definitions"),

    cell("markdown", r"""
    ### 13a · Explain the 88 missing mask joins

    **Prediction.** If the mask archive lacks complete class directories, the missing
    images should concentrate exactly in those species. If the archive has all 70
    directories, the problem is instead filename alignment.
    """, "coverage_header"),

    cell("code", r"""
    mask_archive = CURATED/"cub70"/"masks"/"AnnotationMasksPerclass"
    if not mask_archive.is_dir():
        mask_archive = CURATED/"cub70"/"masks"

    archive_ids = sorted(
        int(p.name.split(".")[0])
        for p in mask_archive.iterdir()
        if p.is_dir() and p.name.split(".")[0].isdigit()
    )
    missing_archive_ids = sorted(set(range(1, 71)) - set(archive_ids))
    all_stems = set(E70.image.unique())
    mask_stems = set(RAWVIS.image_name.unique())
    missing_stems = all_stems - mask_stems
    missing_rows = (
        E70[E70.image.isin(missing_stems)][["image", "y_true"]]
        .drop_duplicates().groupby("y_true").size().rename("missing_images")
        .reset_index()
    )

    classes_path = CURATED/"CUB_200_2011"/"classes.txt"
    class_names = {}
    if classes_path.exists():
        for line in classes_path.read_text().splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2:
                class_names[int(fields[0]) - 1] = fields[1]
    missing_rows["official_class_id"] = missing_rows.y_true + 1
    missing_rows["species_name"] = missing_rows.y_true.map(class_names).fillna("name unavailable")

    print("mask archive class directories:", len(archive_ids))
    print("missing archive class IDs (1-based):", missing_archive_ids)
    print("prediction images without any mask stem:", len(missing_stems))
    display(missing_rows)
    if len(archive_ids) == 67 and missing_archive_ids == [11, 16, 32] \
            and len(missing_stems) == 88 \
            and set(missing_rows.official_class_id) == {11, 16, 32}:
        print("[COVERAGE EXPLAINED] all 88 missing photographs belong to the three "
              "class directories absent from the downloaded mask archive.")
    else:
        print("[COVERAGE STILL OPEN] archive and join omissions do not match exactly.")
    """, "coverage_code"),

    cell("markdown", r"""
    ### 13b · Name every collapsed exact concept

    `unique_prob_6dp <= 2` means the model emitted at most two distinct probabilities
    on positive examples. Such a slot cannot be used as grounding evidence.
    """, "collapse_header"),

    cell("code", r"""
    COLLAPSED = (
        ordinary.loc[ordinary.unique_prob_6dp <= 2,
                     ["attribute_type", "concept_name", "positive_recall",
                      "positive_prob_mean", "positive_prob_std", "unique_prob_6dp"]]
        .sort_values(["attribute_type", "concept_name"])
    )
    display(COLLAPSED)
    print("collapsed exact concepts:", len(COLLAPSED))
    """, "collapse_code"),

    cell("markdown", r"""
    ### 13c · Put the CUB conflict result into bars before relating it to FunnyBirds

    The left bar is a **data rate**: among positive labels, how often is the mask absent?
    The right bar is a **model rate**: among those hidden positive cases, how often does
    the model still predict positive? These are paired summaries, not a causal relabeling
    effect.

    FunnyBird RLv2's intervention-size bars counted changed **training labels**:
    tail `7,489`, beak `367`, eye `268`, foot `48`, wing `12`. CUB70 masks are test-only,
    so the CUB bars below do not claim that any CUB training label was changed.
    """, "conflict_header"),

    cell("code", r"""
    conflict_bar = (EXACT70.groupby("attribute_type")
                    .agg(label_hidden_rate=("label_hidden_rate", "mean"),
                         hidden_positive_rate=("hidden_violation", "mean"),
                         n_exact_concepts=("concept_name", "size"))
                    .sort_values("label_hidden_rate", ascending=False))
    display(conflict_bar.round(3))
    x = np.arange(len(conflict_bar)); w = .42
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x-w/2, conflict_bar.label_hidden_rate, w, label="data: positive label but mask absent")
    ax.bar(x+w/2, conflict_bar.hidden_positive_rate, w, label="model: still predicts positive while mask absent")
    ax.set_xticks(x); ax.set_xticklabels(conflict_bar.index, rotation=70, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("fraction")
    ax.set_title("CUB: label/mask conflict and hidden prediction, summarized as bars")
    ax.legend(); plt.tight_layout(); plt.show()
    """, "conflict_code"),

    cell("markdown", r"""
    ### 13d · Direct analogue of the FunnyBird species probe

    We train a small diagnostic classifier on held-out CUB photographs. For each mask
    block it tries to predict species from either the true concept labels or the CBM's
    concept probabilities. High accuracy means species information is stored in that
    block. It does not identify the pixels used to produce the block.

    Unlike FunnyBirds, CUB blocks have very different numbers of concepts and labels can
    be multi-valued. Therefore there is no simple `n_variants/70` theoretical line; the
    true-label probe is the fair availability baseline.
    """, "probe_header"),

    cell("code", r"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    probe_images = J70[["image", "y_true"]].drop_duplicates().sort_values("image")
    train_ids, test_ids = train_test_split(
        probe_images.image, test_size=.30, random_state=20260731,
        stratify=probe_images.y_true,
    )

    def species_probe(frame, block, value):
        d = frame[frame.mask_group == block]
        X = d.pivot_table(index="image", columns="concept_name", values=value, aggfunc="first").fillna(0)
        y = d[["image", "y_true"]].drop_duplicates().set_index("image").loc[X.index, "y_true"]
        tr = X.index.isin(train_ids); te = X.index.isin(test_ids)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, C=1.0, multi_class="auto", random_state=20260731,
        ))
        model.fit(X.loc[tr], y.loc[tr])
        return accuracy_score(y.loc[te], model.predict(X.loc[te])), X.shape[1]

    probe_rows = []
    for block in sorted(J70.mask_group.dropna().unique()):
        for value, label in [("gt_label", "true concept labels"), ("prob", "CBM probabilities")]:
            acc, dims = species_probe(J70, block, value)
            probe_rows.append({"mask_block": block, "features": label,
                               "species_accuracy": acc, "n_concepts": dims})
    CUB_SPECIES_PROBE = pd.DataFrame(probe_rows)
    display(CUB_SPECIES_PROBE.round(3))
    P = CUB_SPECIES_PROBE.pivot(index="mask_block", columns="features", values="species_accuracy")
    P.plot.bar(figsize=(9, 4), color=["#0072B2", "#999999"])
    plt.axhline(1/probe_images.y_true.nunique(), color="black", ls="--", label="chance")
    plt.ylabel("held-out species accuracy"); plt.ylim(0, 1)
    plt.title("CUB species decoded from each concept block")
    plt.legend(); plt.tight_layout(); plt.show()
    """, "probe_code"),

    cell("markdown", r"""
    ### 13e · Display the unexplained CUB photographs and every available mask

    Four diagnostic cases are selected by declared rules, not by hand:

    1. high label/mask conflict and high hidden prediction;
    2. high conflict but low hidden prediction;
    3. strongest positive visible-minus-hidden effect among noncollapsed concepts;
    4. strongest negative effect among noncollapsed concepts.

    For each selected concept, the figure shows one hidden positive photograph and, when
    available in the same species, one visible positive photograph. The overlay contains
    every released fine mask present in that image. A missing target mask therefore
    remains visibly absent rather than being replaced by a fabricated region.
    """, "examples_header"),

    cell("code", r"""
    from PIL import Image
    import matplotlib.patches as mpatches
    import matplotlib.image as mpimg

    eligible = EXACT70[
        (EXACT70.n_hidden >= 10) & (EXACT70.n_visible >= 10) &
        (EXACT70.unique_prob_6dp > 2)
    ].copy()
    high_conflict = eligible[eligible.label_hidden_rate >= .50]
    if len(high_conflict) < 2:
        high_conflict = eligible.nlargest(max(2, len(eligible)//5), "label_hidden_rate")

    selections = []
    hh = high_conflict.assign(score=high_conflict.label_hidden_rate*high_conflict.hidden_violation).nlargest(1, "score").iloc[0]
    hl = high_conflict.sort_values(["hidden_violation", "label_hidden_rate"], ascending=[True, False]).iloc[0]
    pos = eligible.nlargest(1, "visibility_effect").iloc[0]
    neg = eligible.nsmallest(1, "visibility_effect").iloc[0]
    for case, row in [("high conflict + high violation", hh),
                      ("high conflict + low violation", hl),
                      ("strong positive visibility response", pos),
                      ("negative visibility response", neg)]:
        selections.append({"case": case, **row.to_dict()})
    SELECTED_CASES = pd.DataFrame(selections)
    display(SELECTED_CASES[["case", "attribute_type", "concept_name", "mask_group",
                            "label_hidden_rate", "hidden_violation", "visibility_effect",
                            "n_visible", "n_hidden", "unique_prob_6dp"]].round(3))

    image_root = CURATED/"CUB_200_2011"/"images"
    image_lookup = {p.stem: p for p in image_root.rglob("*.jpg")}
    mask_colors = {part: plt.cm.tab20(i/20) for i, part in enumerate(CUB70_PARTS)}

    def choose_pair(sel):
        d = J70[(J70.concept_name == sel.concept_name) & (J70.gt_label == 1)].copy()
        hidden = d[~d.visible].sort_values("prob", ascending=False if "high violation" in sel.case else True)
        visible = d[d.visible]
        if hidden.empty:
            return None, None
        h = hidden.iloc[0]
        same = visible[visible.y_true == h.y_true]
        if same.empty:
            same = visible
        if same.empty:
            return h, None
        if "negative" in sel.case:
            v = same.sort_values("prob").iloc[0]
        else:
            v = same.sort_values("prob", ascending=False).iloc[0]
        return h, v

    def overlay_for(stem):
        path = image_lookup.get(stem)
        if path is None:
            return None, None, []
        rgb = np.asarray(Image.open(path).convert("RGB"))
        overlay = rgb.astype(float)/255
        class_rows = RAWVIS[RAWVIS.image_name == stem]
        if class_rows.empty:
            return rgb, overlay, []
        cid = int(class_rows.class_idx.iloc[0]) + 1
        class_dir = mask_archive/str(cid)
        present = []
        for part in CUB70_PARTS:
            f = class_dir/f"{stem}_{part}.png"
            if not f.exists():
                continue
            mask = np.asarray(Image.open(f).convert("L")) > 0
            if mask.shape != rgb.shape[:2]:
                mask = np.asarray(Image.fromarray(mask.astype("uint8")*255).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 0
            color = np.array(mask_colors[part][:3])
            overlay[mask] = .40*overlay[mask] + .60*color
            present.append(part)
        return rgb, overlay, present

    pairs = [(row, *choose_pair(row)) for row in SELECTED_CASES.itertuples()]
    fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 3.6*len(pairs)))
    for r, (sel, hidden, visible) in enumerate(pairs):
        for pair_col, record, state in [(0, hidden, "target mask absent"),
                                        (2, visible, "target mask visible")]:
            for offset in [0, 1]: axes[r, pair_col+offset].axis("off")
            if record is None:
                axes[r, pair_col].text(.5, .5, "no matching example", ha="center", va="center")
                continue
            rgb, overlay, present = overlay_for(record.image)
            if rgb is None:
                axes[r, pair_col].text(.5, .5, f"image missing\n{record.image}", ha="center", va="center")
                continue
            axes[r, pair_col].imshow(rgb)
            axes[r, pair_col+1].imshow(overlay)
            axes[r, pair_col].set_title(
                f"{sel.case}\n{state}\n{record.image} · species {record.y_true}\n"
                f"{sel.concept_name} · probability={record.prob:.3f}", fontsize=8)
            axes[r, pair_col+1].set_title(
                f"all available masks\ntarget block={sel.mask_group}\n"
                f"present: {', '.join(present) if present else 'none'}", fontsize=8)
    handles = [mpatches.Patch(color=mask_colors[p], label=p) for p in CUB70_PARTS]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    plt.tight_layout(rect=(0, .04, 1, 1)); plt.show()
    """, "examples_code"),

    cell("markdown", r"""
    ### Interpretation rule for the example grid

    The images can diagnose obvious annotation or pose problems, but one photograph
    cannot prove a population mechanism.

    - Target pixels genuinely absent, other masks plausible, high score: consistent
      with contextual prediction.
    - Target visibly present in RGB but mask absent: mask/annotation failure.
    - Extreme crop, profile, or overlap: pose confound.
    - Concept listed in the collapse table: collapsed output, not grounding evidence.
    - No obvious visual problem across several cases: motivates a controlled CUB
      intervention; it still does not turn natural visibility into a causal swap.
    """, "examples_rule"),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [
        c for c in nb["cells"]
        if c.get("metadata", {}).get("cub05_section") != SECTION_ID
    ]
    nb["cells"].extend(CELLS)
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"inserted {len(CELLS)} comparative-audit cells into {NOTEBOOK}")


if __name__ == "__main__":
    main()
