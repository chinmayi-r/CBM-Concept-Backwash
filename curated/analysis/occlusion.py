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


def relabel_flip_summary(diag_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize relabel_cub_with_cub70 diagnostics: flips per part."""
    g = (diag_df.groupby("part")
         .agg(considered=("flipped", "size"),
              flipped=("flipped", "sum"))
         .reset_index())
    g["flip_rate"] = g["flipped"] / g["considered"].clip(lower=1)
    return g.sort_values("flip_rate", ascending=False)


def condition_comparison(tables: dict, vis_df: pd.DataFrame,
                         prob_thresh=0.5) -> pd.DataFrame:
    """Build the prof-note-#5 verdict table across conditions.

    `tables`: {condition_name: eval_df}. Returns one row per condition with
    overall task acc, mean concept acc, and overall grounding violation rate."""
    rows = []
    for name, ev in tables.items():
        joined = attach_visibility(ev, vis_df)
        vr = grounding_violation_rate(joined, prob_thresh)
        img = ev.drop_duplicates("image")
        rows.append({
            "condition": name,
            "task_acc": (img.y_true == img.y_pred).mean(),
            "concept_acc": (ev.gt_label == ev.pred_label).mean(),
            "violation_rate": float(np.average(
                vr.violation_rate, weights=vr.n_occluded)) if len(vr) else np.nan,
        })
    return pd.DataFrame(rows)
