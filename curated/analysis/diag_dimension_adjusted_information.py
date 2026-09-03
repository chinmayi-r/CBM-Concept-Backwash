"""D6.1 — species information in raw scores after binary labels are known.

The primary paired comparison is labels-only versus labels plus within-label
raw-score residuals. Raw-only connects to notebook Figure 8b; it is not called
conditional information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diag_common as dc


def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=1.0, random_state=dc.FOLD_SEED),
    )


def _within_label_residual(z_tr, c_tr, z_te, c_te):
    """Subtract training-fold E[z_j | c_j] without looking at test scores."""
    train = np.empty_like(z_tr, dtype=float)
    test = np.empty_like(z_te, dtype=float)
    for j in range(z_tr.shape[1]):
        fallback = float(np.mean(z_tr[:, j]))
        means = {}
        for label in (0, 1):
            selected = z_tr[c_tr[:, j] == label, j]
            means[label] = float(np.mean(selected)) if len(selected) else fallback
        train[:, j] = z_tr[:, j] - np.array([means[int(v)] for v in c_tr[:, j]])
        test[:, j] = z_te[:, j] - np.array([means[int(v)] for v in c_te[:, j]])
    return train, test


def compare_cv(z: np.ndarray, c: np.ndarray, y: np.ndarray):
    """OOF predictions for labels, raw scores, and labels+score residuals."""
    classes = np.unique(y)
    splitter = StratifiedKFold(
        n_splits=dc.N_FOLDS, shuffle=True, random_state=dc.FOLD_SEED)
    predictions = {
        name: np.full((len(y), len(classes)), np.nan)
        for name in ("labels", "raw", "labels_plus_residual")
    }
    for tr, te in splitter.split(z, y):
        residual_tr, residual_te = _within_label_residual(
            z[tr], c[tr], z[te], c[te])
        inputs = {
            "labels": (c[tr].astype(float), c[te].astype(float)),
            "raw": (z[tr], z[te]),
            "labels_plus_residual": (
                np.column_stack([c[tr], residual_tr]),
                np.column_stack([c[te], residual_te]),
            ),
        }
        for name, (x_tr, x_te) in inputs.items():
            model = _model()
            model.fit(x_tr, y[tr])
            fold_probability = model.predict_proba(x_te)
            aligned = np.full((len(te), len(classes)), 1e-12)
            positions = {species: k for k, species in enumerate(classes)}
            for source_col, species in enumerate(model.classes_):
                aligned[:, positions[species]] = fold_probability[:, source_col]
            aligned /= aligned.sum(axis=1, keepdims=True)
            predictions[name][te] = aligned
    if any(np.isnan(value).any() for value in predictions.values()):
        raise RuntimeError("D6.1 out-of-fold predictions are incomplete")
    return classes, predictions


def summarize(z, c, y, with_interval=True):
    classes, pred = compare_cv(z, c, y)
    species_col = {species: k for k, species in enumerate(classes)}
    truth_col = np.array([species_col[species] for species in y])
    losses = {
        name: -np.log(np.clip(prob[np.arange(len(y)), truth_col], 1e-12, 1.0))
        for name, prob in pred.items()
    }
    conditional_delta = losses["labels"] - losses["labels_plus_residual"]
    gain = float(np.mean(conditional_delta))
    gain_lo = gain_hi = np.nan
    if with_interval:
        gain, gain_lo, gain_hi = dc.clustered_metric_interval(
            conditional_delta, np.arange(len(y)), np.mean)
    result = {
        "labels_logloss": float(np.mean(losses["labels"])),
        "raw_only_logloss": float(np.mean(losses["raw"])),
        "labels_plus_residual_logloss": float(np.mean(losses["labels_plus_residual"])),
        "conditional_logloss_gain": gain,
        "conditional_gain_ci_low": gain_lo,
        "conditional_gain_ci_high": gain_hi,
    }
    for name, prob in pred.items():
        result[f"{name}_accuracy"] = accuracy_score(y, classes[prob.argmax(axis=1)])
    return result


def main():
    z, c, y, _, _ = dc.load_eval()
    _, spans = dc.load_concepts()
    out = dc.out_dir()
    rng = np.random.default_rng(dc.SUBSET_SEED)

    rows = []
    for part in dc.ORDER:
        lo, hi = spans[part]
        result = summarize(z[:, lo:hi], c[:, lo:hi].astype(int), y)
        result.update({"part": part, "K": hi - lo})
        result["gain_per_dimension_descriptive"] = (
            result["conditional_logloss_gain"] / (hi - lo))
        rows.append(result)
    table = pd.DataFrame(rows)

    # Sensitivity only: equal coordinate count does not equalize value cardinality,
    # frequency, geometry, or species structure.
    lo, hi = spans["tail"]
    subset_rows = []
    for repeat in range(40):
        local = np.sort(rng.choice(np.arange(hi - lo), size=3, replace=False))
        cols = lo + local
        result = summarize(z[:, cols], c[:, cols].astype(int), y, with_interval=False)
        subset_rows.append({
            "repeat": repeat,
            "coordinates": ",".join(map(str, local)),
            "conditional_logloss_gain": result["conditional_logloss_gain"],
        })
    subsets = pd.DataFrame(subset_rows)
    eye_gain = float(table.loc[table.part == "eye", "conditional_logloss_gain"].iloc[0])
    subset_summary = pd.DataFrame([{
        "test": "tail on 3 random of 9 coordinates (sensitivity only)",
        "repeats": len(subsets),
        "mean_conditional_gain": subsets.conditional_logloss_gain.mean(),
        "min_conditional_gain": subsets.conditional_logloss_gain.min(),
        "max_conditional_gain": subsets.conditional_logloss_gain.max(),
        "eye_K3_conditional_gain_reference": eye_gain,
        "interpretation": "equalizes coordinate count only; not cardinality or species structure",
    }])

    table.round(5).to_csv(out / "d61_dimension_adjusted_information.csv", index=False)
    subsets.round(5).to_csv(out / "d61_tail_subset_repeats.csv", index=False)
    subset_summary.round(5).to_csv(out / "d61_tail_subset_sensitivity.csv", index=False)
    print("\nD6.1 · held-out species information after binary labels are already known")
    print(table.round(4).to_string(index=False))
    print("\nD6.1 · three-coordinate tail sensitivity")
    print(subset_summary.round(4).to_string(index=False))
    print("\nReading rule: conditional_logloss_gain compares the same held-out image "
          "under labels-only and labels+within-label residual scores. Positive means "
          "raw magnitudes add species information after labels are known. The 95% "
          "interval resamples images and does not represent training-seed uncertainty. "
          "The 3-coordinate tail test addresses width only and cannot by itself explain "
          "tail backwash.")


if __name__ == "__main__":
    main()
