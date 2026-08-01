#!/usr/bin/env python3
"""Append discriminating tests motivated by visual inspection of CUB Figure 17."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "05_cub_cbm.ipynb"
SECTION = "cub05_discriminating_tests"


def src(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def md(text: str, name: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"cub05_section": SECTION, "cub05_cell": name}, "source": src(text)}


def code(text: str, name: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"cub05_section": SECTION, "cub05_cell": name}, "source": src(text)}


CELLS = [
    md(r"""
    ## 14 · Discriminating tests prompted by the inspected photographs

    Figure 17 did not yield a clean contextual-prediction example. Several regions were
    visibly present in RGB even when the released mask was absent, and the strongest
    positive/negative pairs also changed scale and pose. We therefore test the simpler
    explanations before using those examples as evidence of backwash.
    """, "header"),

    md(r"""
    ### 14a · Are the remaining seven mask joins archive omissions or spelling errors?

    The three absent class directories explain 81 images. Print the other seven stems
    and search the mask archive for normalized or near-matching names. No nearby file
    means the archive genuinely omits those individual photographs.
    """, "missing_header"),

    code(r"""
    import difflib, re

    missing_detail = (
        E70[E70.image.isin(missing_stems)][["image", "y_true"]]
        .drop_duplicates().assign(official_class_id=lambda d: d.y_true + 1)
    )
    missing_detail["species_name"] = missing_detail.y_true.map(class_names).fillna("name unavailable")
    extra = missing_detail[~missing_detail.official_class_id.isin(missing_archive_ids)].copy()

    def norm_stem(x):
        return re.sub(r"[^a-z0-9]", "", str(x).lower())

    archive_stems = sorted(mask_stems)
    norm_lookup = {}
    for stem in archive_stems:
        norm_lookup.setdefault(norm_stem(stem), []).append(stem)
    rows = []
    for r in extra.itertuples():
        exact_normalized = norm_lookup.get(norm_stem(r.image), [])
        near = difflib.get_close_matches(r.image, archive_stems, n=3, cutoff=.82)
        rows.append({
            "image": r.image, "official_class_id": r.official_class_id,
            "species_name": r.species_name,
            "normalized_matches": exact_normalized, "nearest_archive_stems": near,
        })
    EXTRA_MISSING = pd.DataFrame(rows)
    display(EXTRA_MISSING)
    if len(EXTRA_MISSING) == 7 and not EXTRA_MISSING.normalized_matches.map(bool).any():
        print("[SEVEN INDIVIDUAL OMISSIONS] no normalized filename match exists in the parsed archive.")
    else:
        print("[JOIN INVESTIGATION NEEDED] a filename variant may explain at least one row.")
    """, "missing_code"),

    md(r"""
    ### 14b · Separate zero annotated pixels from a tiny annotated region

    `mask absent` previously combined two cases:

    - exactly zero pixels in the released masks;
    - a nonzero mask smaller than the 0.1% visibility threshold.

    If a result comes mainly from tiny nonzero masks, the threshold is important. If it
    comes from zero-pixel rows whose region is visibly present in RGB, mask completeness
    or attribute-to-mask mismatch is the first explanation.
    """, "mask_state_header"),

    code(r"""
    POS = J70[J70.gt_label == 1].copy()
    POS["mask_state"] = np.select(
        [POS.pixel_count == 0, (POS.pixel_count > 0) & (~POS.visible)],
        ["zero annotated pixels", "tiny nonzero mask"], default="visible",
    )
    MASK_STATE = (POS.groupby(["attribute_type", "mask_state"])
                  .agg(n=("image", "size"), mean_probability=("prob", "mean"),
                       predicted_positive=("pred_label", "mean"))
                  .reset_index())
    display(MASK_STATE.round(3))
    state_counts = MASK_STATE.pivot(index="attribute_type", columns="mask_state", values="n").fillna(0)
    state_counts = state_counts.div(state_counts.sum(axis=1), axis=0)
    state_counts.plot.bar(stacked=True, figsize=(13, 4),
                          color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.ylabel("fraction of positive-labelled rows"); plt.ylim(0, 1)
    plt.title("CUB positive labels: zero pixels, tiny mask, or visible mask")
    plt.tight_layout(); plt.show()
    """, "mask_state_code"),

    md(r"""
    ### 14c · CUB correct-versus-best-wrong margin

    This is the closest available CUB analogue to FunnyBird donor-versus-source
    ordering. For each image and attribute type with at least one selected positive and
    one selected negative value, define:

    `contrast = highest probability among correct values - highest probability among incorrect values`.

    Positive contrast means a correct value beats the strongest selected wrong value.
    We compare this contrast between naturally visible and absent masks. Unlike the
    FunnyBird swap margin, it remains observational because these are different images.
    """, "contrast_header"),

    code(r"""
    contrast_rows = []
    for (image, atype), d in J70.groupby(["image", "attribute_type"]):
        pos = d.loc[d.gt_label == 1, "prob"]
        neg = d.loc[d.gt_label == 0, "prob"]
        if pos.empty or neg.empty:
            continue
        contrast_rows.append({
            "image": image, "attribute_type": atype, "mask_group": d.mask_group.iloc[0],
            "y_true": int(d.y_true.iloc[0]), "visible": bool(d.visible.iloc[0]),
            "pixel_count": int(d.pixel_count.iloc[0]),
            "correct_best": float(pos.max()), "wrong_best": float(neg.max()),
            "contrast": float(pos.max() - neg.max()),
        })
    CONTRAST = pd.DataFrame(contrast_rows)

    raw_contrast = (CONTRAST.groupby(["attribute_type", "visible"]).contrast.mean()
                    .unstack("visible").dropna())
    raw_contrast["visibility_contrast_effect"] = raw_contrast[True] - raw_contrast[False]

    matched = (CONTRAST.groupby(["attribute_type", "y_true", "visible"]).contrast.mean()
               .unstack("visible").dropna())
    matched["effect"] = matched[True] - matched[False]
    matched_summary = (matched.groupby("attribute_type").effect
                       .agg(["mean", "median", "count"]).sort_values("mean", ascending=False))
    display(raw_contrast.round(3))
    display(matched_summary.round(3))

    order = matched_summary.index
    vals = [matched.loc[t, "effect"].values if t in matched.index.get_level_values(0) else []
            for t in order]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.boxplot(vals, tick_labels=order, showfliers=False)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticklabels(order, rotation=70, ha="right")
    ax.set_ylabel("visible - absent change in correct-vs-best-wrong contrast")
    ax.set_title("CUB concept ordering after matching species (observational)")
    plt.tight_layout(); plt.show()
    """, "contrast_code"),

    md(r"""
    ### Decision rule

    - Positive visibility effect **and** positive contrast effect: visibility raises the
      score and helps the correct value beat alternatives.
    - Positive visibility effect but flat contrast: all values may move together; this
      is sensitivity without specific grounding.
    - Negative contrast effect driven by zero-pixel yet visibly present examples: inspect
      mask quality before naming backwash.
    - Negative effect surviving species matching, noncollapsed outputs, adequate sample
      size, and visually valid masks: candidate contextual prediction, still requiring a
      controlled image intervention for causal proof.
    """, "decision"),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [c for c in nb["cells"] if c.get("metadata", {}).get("cub05_section") != SECTION]
    nb["cells"].extend(CELLS)
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"inserted {len(CELLS)} discriminating-test cells into {NOTEBOOK}")


if __name__ == "__main__":
    main()
