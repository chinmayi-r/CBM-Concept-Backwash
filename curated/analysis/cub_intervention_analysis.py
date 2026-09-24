"""Outcome accounting for a reviewed CUB edit experiment and matched label test.

This module does not generate edits, accept a segmentation model, or establish
that an edit is causal. It keeps those inputs explicit and reports unsuccessful
responses rather than conditioning the whole experiment on donor recognition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_reviewed_results(root):
    """Join a blinded review to immutable frozen-inference predictions."""
    import hashlib
    import json
    from pathlib import Path
    root = Path(root)
    manifest = json.loads((root / "INFERENCE.json").read_text())
    predictions = root / "predictions_unreviewed.parquet"
    if manifest.get("framework") != "koh_joint" or manifest.get("training") is not False:
        raise ValueError("Expected recorded frozen Koh Joint inference")
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != manifest.get("predictions_sha256"):
        raise ValueError("Frozen predictions changed after inference")
    frame = pd.read_parquet(predictions)
    review = pd.read_csv(root / "blinded_review.csv")
    keys = ["trial_id", "seed"]
    if review.duplicated(keys).any() or set(review) != set(keys + REVIEWS):
        raise ValueError("Blinded review must contain one exact record per trial/seed")
    merged = frame.drop(columns=REVIEWS).merge(review, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Review and prediction populations differ")
    return score_interventions(merged.drop(columns="_merge")), manifest

SCORES = [f"z_{concept}_{condition}" for concept in ("source", "donor")
          for condition in ("original", "same_value", "different_value")]
IDENTITY = ["trial_id", "source_image", "same_value_donor_image", "different_value_donor_image",
            "attribute_type", "source_concept", "donor_concept", "mask_group", "seed",
            "source_pixels_sha256", "same_value_pixels_sha256", "different_value_pixels_sha256"]
REVIEWS = ["input_identity_verified", "edit_footprint_verified", "target_value_reviewed",
           "other_regions_preserved", "review_blinded_to_logits"]


def score_interventions(frame):
    required = set(IDENTITY + REVIEWS + SCORES)
    if required - set(frame):
        raise ValueError(f"Missing intervention fields: {sorted(required - set(frame))}")
    if frame.empty or frame[IDENTITY].isna().any().any():
        raise ValueError("Intervention identities must be complete and nonempty")
    if frame.duplicated(["trial_id", "seed"]).any():
        raise ValueError("Duplicate trial/seed records")
    if (frame.source_concept == frame.donor_concept).any():
        raise ValueError("Source and different-value donor concepts must differ")
    if (frame.same_value_donor_image == frame.different_value_donor_image).any():
        raise ValueError("Same-value and different-value donors must be separate identified images")
    if not frame.mask_group.isin(["head", "eye", "beak", "neck", "body", "wing", "leg", "tail"]).all():
        raise ValueError("Unrecognized CUB anatomical group")
    for name in REVIEWS:
        if not frame[name].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError(f"{name} must contain explicit booleans, not strings or missing values")
    for name in [k for k in IDENTITY if k.endswith("sha256")]:
        if not frame[name].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid pixel hash in {name}")
    values = frame[SCORES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite intervention scores")
    out = frame.copy()
    for condition in ("original", "same_value", "different_value"):
        out[f"margin_{condition}"] = out[f"z_donor_{condition}"] - out[f"z_source_{condition}"]
    out["response_delta"] = out.margin_different_value - out.margin_original
    out["control_adjusted_response"] = out.margin_different_value - out.margin_same_value
    out["same_value_edit_drift"] = out.margin_same_value - out.margin_original
    out["review_eligible"] = out[REVIEWS].all(axis=1)
    out["donor_wins"] = out.margin_different_value > 0
    out["positive_response_source_retained"] = (out.response_delta > 0) & (out.margin_different_value < 0)
    out["nonpositive_response_source_retained"] = (out.response_delta <= 0) & (out.margin_different_value < 0)
    out["margin_tie"] = out.margin_different_value == 0
    out["control_adjusted_positive_source_retained"] = (out.control_adjusted_response > 0) & (out.margin_different_value < 0)
    out["different_edit_unchanged"] = out.different_value_pixels_sha256 == out.source_pixels_sha256
    if (out.review_eligible & out.different_edit_unchanged).any():
        raise ValueError("A reviewed insertion has unchanged source pixels")
    return out


def summarize_interventions(scored):
    rows = []
    for (seed, part), group in scored.groupby(["seed", "mask_group"], sort=True):
        eligible = group[group.review_eligible]
        n = len(eligible)
        record = {"seed": seed, "mask_group": part, "selected_trials": len(group),
                  "review_eligible_trials": n, "excluded_trials": len(group) - n,
                  "source_images": eligible.source_image.nunique(),
                  "interpretation": "paired edit diagnostic; causal acceptance requires method calibration"}
        for name in ["donor_wins", "positive_response_source_retained", "nonpositive_response_source_retained",
                     "margin_tie", "control_adjusted_positive_source_retained"]:
            record[name + "_count"] = int(eligible[name].sum())
            record[name + "_fraction"] = float(eligible[name].mean()) if n else np.nan
        for name in ["margin_original", "margin_different_value", "response_delta",
                     "control_adjusted_response", "same_value_edit_drift"]:
            record["median_" + name] = float(eligible[name].median()) if n else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def compare_relabeling(standard, relabeled):
    """Compare frozen predictions on identical edited pixels, one trained seed at a time.

    Models/training manifests must separately establish an otherwise matched
    intervention. No image-level confidence interval is a seed-replication CI.
    """
    a, b = score_interventions(standard), score_interventions(relabeled)
    keys = ["trial_id", "seed"]
    a = a.set_index(keys).sort_index()
    b = b.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Standard/RLv2 must contain exactly the same trial/seed set")
    compare = [c for c in IDENTITY if c not in keys] + REVIEWS
    if not a[compare].equals(b[compare]):
        raise ValueError("Standard/RLv2 identity, edited pixels, or blinded review differs")
    out = a[compare].copy()
    out["review_eligible"] = a.review_eligible
    for name in ["response_delta", "margin_different_value", "control_adjusted_response"]:
        out[name + "_standard"] = a[name]
        out[name + "_relabeled"] = b[name]
        out[name + "_change"] = b[name] - a[name]
    out["retention_resolved"] = a.positive_response_source_retained & ~b.positive_response_source_retained
    out["retention_introduced"] = ~a.positive_response_source_retained & b.positive_response_source_retained
    # Resolution alone can reflect loss of response, so explicitly separate it.
    out["resolved_with_donor_win"] = a.positive_response_source_retained & b.donor_wins
    out["resolved_by_losing_response"] = a.positive_response_source_retained & b.nonpositive_response_source_retained
    return out.reset_index()


def joint_logit_gradient(logits, weights, bias, label, concepts, concept_weights, attr_weight=.01):
    """Per-example main-output Koh Joint gradient; auxiliary branch excluded explicitly.

    Koh uses BCE weight, not pos_weight. The scalar concept weight multiplies
    both positive and negative BCE terms. Normalization is 1 + attr_weight * J.
    """
    z = np.asarray(logits, dtype=float)
    w, b = np.asarray(weights, dtype=float), np.asarray(bias, dtype=float)
    c, q = np.asarray(concepts, dtype=float), np.asarray(concept_weights, dtype=float)
    if z.ndim != 1 or w.shape != (len(b), len(z)) or c.shape != z.shape or q.shape != z.shape:
        raise ValueError("Incompatible model/label dimensions")
    if not all(np.isfinite(v).all() for v in [z, w, b, c, q]) or (q < 0).any():
        raise ValueError("Gradient inputs must be finite with nonnegative BCE weights")
    if not np.isin(c, [0, 1]).all() or not 0 <= label < len(b):
        raise ValueError("Invalid labels")
    scores = w @ z + b
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    error = probs.copy()
    error[label] -= 1
    sigmoid = np.exp(-np.logaddexp(0, -z))
    denom = 1 + attr_weight * len(z)
    task = w.T @ error / denom
    concept = attr_weight * q * (sigmoid - c) / denom
    return {"task_gradient": task, "concept_gradient": concept, "total_gradient": task + concept}
