"""Occlusion / grounding analysis, ported from the FunnyBirds template.

Central question (prof notes #2/#4): is a concept's z near 1 even when the part
is occluded? If z stays high regardless of visibility, the bottleneck is not
grounded in visible evidence for that concept.

All functions take the normalized eval table (io.EVAL_SCHEMA) and a visibility
table with columns [image, part, area_frac, visible]; they join on (image, part).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


def attach_visibility(eval_df: pd.DataFrame, vis_df: pd.DataFrame) -> pd.DataFrame:
    """Join per-(image,part) visibility onto the eval table.

    `vis_df` must have coarse parts matching eval_df.part. For CUB70 use the
    coarse collapse (see data/cub70/relabel_cub_with_cub70.coarse_visibility)."""
    v = vis_df.rename(columns={"image_name": "image"})[
        ["image", "part", "area_frac", "visible"] if "area_frac" in vis_df.columns
        else ["image", "part", "visible"]
    ]
    # guard against a double-join: drop any visibility columns already on eval_df
    # so the merge can't silently produce visible_x/visible_y.
    eval_df = eval_df.drop(columns=[c for c in ("visible", "area_frac")
                                    if c in eval_df.columns], errors="ignore")
    out = eval_df.merge(v, on=["image", "part"], how="inner")
    return out


def z_by_visibility(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median z and prob split by visible vs occluded, per part.

    Only rows whose GT label says the concept is present are kept: those are the
    images where CUB *claims* the part is there, so disagreement with the mask is
    exactly the species-constant-label artifact we are probing."""
    present = df[df.gt_label == 1]
    g = (present.groupby(["part", "visible"])
         .agg(n=("z", "size"), z_mean=("z", "mean"), z_median=("z", "median"),
              prob_mean=("prob", "mean"))
         .reset_index())
    return g


def quartile_grounding(df: pd.DataFrame, value="area_frac", q=4) -> pd.DataFrame:
    """Bin present-labeled concepts into visibility quartiles and report mean prob.

    This is the CUB analogue of the FunnyBirds Q1..Q4 pixel-count analysis: if
    mean prob rises monotonically with visibility quartile, z is (partly) grounded;
    if it is flat and high, it is anchoring on a species prior."""
    present = df[(df.gt_label == 1) & df[value].notna()].copy()
    if present.empty:
        return pd.DataFrame()
    present["qbin"] = present.groupby("part")[value].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=min(q, s.nunique()),
                          labels=False, duplicates="drop") + 1)
    return (present.groupby(["part", "qbin"])
            .agg(n=("prob", "size"), prob_mean=("prob", "mean"),
                 z_mean=("z", "mean"), area_frac_mean=(value, "mean"))
            .reset_index())


def grounding_violation_rate(df: pd.DataFrame, prob_thresh=0.5) -> pd.DataFrame:
    """Per part: fraction of OCCLUDED, present-labeled concepts the model still
    predicts present (prob>=thresh). High = ungrounded / anchoring."""
    occ = df[(df.gt_label == 1) & (~df.visible.astype(bool))]
    if occ.empty:
        return pd.DataFrame(columns=["part", "n_occluded", "violation_rate"])
    g = (occ.assign(violate=(occ.prob >= prob_thresh).astype(int))
         .groupby("part")
         .agg(n_occluded=("violate", "size"), violation_rate=("violate", "mean"))
         .reset_index())
    return g


def within_species_visibility_effect(
        df: pd.DataFrame, value: str = "prob") -> pd.DataFrame:
    """Visibility effect after removing the species/concept lookup shortcut.

    The naive visible-vs-occluded comparison may compare different species.
    Because standard CUB labels are species-majority labels, that can manufacture
    an apparent visibility effect (or hide one).  This function first compares
    visible and occluded images *within the same*
    ``(class_label, concept_name)`` group, retaining only groups that contain
    both visibility states.  It then aggregates those paired differences by
    body part.

    A positive ``visible_minus_occluded`` means the concept output is higher
    when its named part is visible even after holding species and concept fixed.
    This controls the species main effect; it still cannot control every pose,
    viewpoint, or background difference in observational photographs.
    """
    required = {"class_label", "concept_name", "part", "visible", "gt_label", value}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"within_species_visibility_effect missing columns: {sorted(missing)}")
    present = df[df.gt_label == 1].copy()
    grouped = (present.groupby(
        ["part", "class_label", "concept_name", "visible"], observed=True)[value]
        .agg(["mean", "size"]).reset_index())
    means = grouped.pivot(
        index=["part", "class_label", "concept_name"],
        columns="visible", values="mean")
    counts = grouped.pivot(
        index=["part", "class_label", "concept_name"],
        columns="visible", values="size")
    if True not in means.columns or False not in means.columns:
        return pd.DataFrame(columns=[
            "part", "n_matched_groups", "n_visible", "n_occluded",
            "visible_mean", "occluded_mean", "visible_minus_occluded",
        ])
    matched = means.dropna(subset=[False, True]).copy()
    if matched.empty:
        return pd.DataFrame(columns=[
            "part", "n_matched_groups", "n_visible", "n_occluded",
            "visible_mean", "occluded_mean", "visible_minus_occluded",
        ])
    matched["difference"] = matched[True] - matched[False]
    matched["n_visible"] = counts.loc[matched.index, True]
    matched["n_occluded"] = counts.loc[matched.index, False]
    matched = matched.reset_index()
    return (matched.groupby("part", observed=True)
            .agg(n_matched_groups=("difference", "size"),
                 n_visible=("n_visible", "sum"),
                 n_occluded=("n_occluded", "sum"),
                 visible_mean=(True, "mean"),
                 occluded_mean=(False, "mean"),
                 visible_minus_occluded=("difference", "mean"))
            .reset_index())


def visibility_specificity_control(df: pd.DataFrame) -> pd.DataFrame:
    """Report visibility effects separately for positive and negative labels.

    The grounding hypothesis concerns ``gt_label=1``: a named positive concept
    should weaken when its part is hidden.  If probabilities move by the same
    amount for ``gt_label=0``, the plot may instead reflect a general pose or
    image-quality effect.  This negative-label row is therefore a specificity
    control, not another backwash metric.
    """
    rows = []
    for (part, label), group in df.groupby(["part", "gt_label"], observed=True):
        visible = group[group.visible.astype(bool)].prob
        occluded = group[~group.visible.astype(bool)].prob
        rows.append({
            "part": part,
            "gt_label": int(label),
            "n_visible": len(visible),
            "n_occluded": len(occluded),
            "prob_visible": visible.mean() if len(visible) else np.nan,
            "prob_occluded": occluded.mean() if len(occluded) else np.nan,
            "visible_minus_occluded": (
                visible.mean() - occluded.mean()
                if len(visible) and len(occluded) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def relabel_flip_summary(diag_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize relabel_cub_with_cub70 diagnostics: flips per part."""
    g = (diag_df.groupby("part")
         .agg(considered=("flipped", "size"),
              flipped=("flipped", "sum"))
         .reset_index())
    g["flip_rate"] = g["flipped"] / g["considered"].clip(lower=1)
    return g.sort_values("flip_rate", ascending=False)


def concept_recall_gap(df: pd.DataFrame, by: str = "part") -> pd.DataFrame:
    """Recall-gap metric (the paper's headline framing of backwash).

    On present-labeled concepts (gt==1), split by visibility and report concept
    recall = P(pred==1 | gt==1) when the part is VISIBLE vs OCCLUDED, and the gap.

        recall_visible  high, recall_occluded low  -> GROUNDED (fires only when seen)
        recall_visible  high, recall_occluded high -> BACKWASH  (fires regardless)
        recall_visible  low                        -> model is just weak on this part

    Note recall_occluded is exactly grounding_violation_rate; the gap is the
    complementary, more legible view. `by`='part' or 'concept_name'.
    """
    present = df[df.gt_label == 1].copy()
    present["hit"] = (present.prob >= 0.5).astype(int)
    present["vis"] = present.visible.astype(bool)
    def _agg(g):
        v, o = g[g.vis], g[~g.vis]
        rv = v.hit.mean() if len(v) else np.nan
        ro = o.hit.mean() if len(o) else np.nan
        return pd.Series({"n_visible": len(v), "n_occluded": len(o),
                          "recall_visible": rv, "recall_occluded": ro,
                          "recall_gap": (rv - ro) if len(v) and len(o) else np.nan})
    return present.groupby(by).apply(_agg, include_groups=False).reset_index()


def counterfactual_deletion(pre: pd.DataFrame, post: pd.DataFrame,
                            part: str, prob_thresh: float = 0.5) -> pd.DataFrame:
    """Causal grounding test via a renderer counterfactual (the FunnyBirds move).

    `pre`  : eval table on the ORIGINAL renders.
    `post` : eval table on renders where `part` was DELETED (same images/ids).
    For the deleted part's concepts that were present & predicted in `pre`, a
    grounded model must now predict ABSENT. We report the fraction still
    predicted present (backwash_rate) and the mean z drop (grounded -> large).

    This is what the correlational occlusion plot cannot show: it rules out
    "legitimate contextual inference" because the rest of the bird is unchanged.
    """
    key = ["image", "concept_idx"]
    a = pre[pre.part == part][key + ["z", "prob", "gt_label"]]
    b = post[post.part == part][key + ["z", "prob"]].rename(
        columns={"z": "z_post", "prob": "prob_post"})
    m = a.merge(b, on=key, how="inner")
    fired = m[(m.gt_label == 1) & (m.prob >= prob_thresh)]
    if fired.empty:
        return pd.DataFrame([{"part": part, "n": 0, "backwash_rate": np.nan,
                              "mean_z_drop": np.nan}])
    return pd.DataFrame([{
        "part": part,
        "n": len(fired),
        "backwash_rate": float((fired.prob_post >= prob_thresh).mean()),
        "mean_z_drop": float((fired.z - fired.z_post).mean()),
    }])


def _shared_eval_images(tables: dict) -> set:
    """Intersection of image sets across conditions (for fair comparison)."""
    sets = [set(ev.image.unique()) for ev in tables.values()]
    return set.intersection(*sets) if sets else set()


def condition_comparison(tables: dict, vis_df: pd.DataFrame,
                         prob_thresh=0.5, restrict_to_shared=True,
                         controlled_pair=None) -> pd.DataFrame:
    """Build the prof-note-#5 verdict table across conditions.

    `tables`: {condition_name: eval_df}. Returns one row per condition with
    overall task acc, mean concept acc, and overall grounding violation rate.

    CONFOUND GUARDS (added after methodological review):
      * restrict_to_shared: score every condition on the SAME images (the
        intersection), so B18 is not apples-to-oranges when conditions were
        trained/evaluated on different subsets. On by default.
      * controlled_pair may mark two genuinely retrained conditions. CUB70 masks
        are test-only, so an original-vs-evaluation-relabeled CUB70 pair is NOT a
        retraining intervention and must not be marked controlled.
    """
    shared = _shared_eval_images(tables) if restrict_to_shared else None
    rows = []
    for name, ev in tables.items():
        if shared is not None:
            ev = ev[ev.image.isin(shared)]
        joined = attach_visibility(ev, vis_df)
        vr = grounding_violation_rate(joined, prob_thresh)
        img = ev.drop_duplicates("image")
        rows.append({
            "condition": name,
            "n_images": img.shape[0],
            "task_acc": (img.y_true == img.y_pred).mean(),
            "concept_acc": (ev.gt_label == ev.pred_label).mean(),
            "violation_rate": float(np.average(
                vr.violation_rate, weights=vr.n_occluded)) if len(vr) else np.nan,
            "controlled": bool(controlled_pair and name in controlled_pair),
        })
    out = pd.DataFrame(rows)
    if restrict_to_shared and shared is not None:
        out.attrs["n_shared_images"] = len(shared)
    return out
