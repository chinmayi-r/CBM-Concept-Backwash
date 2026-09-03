"""D6.1 — Dimension-adjusted conditional species information (Figure 8b follow-up).

Predeclared in DECISIONS.md D6.1. Question: does tail carry an unusually strong
species fingerprint per score coordinate, or does its advantage in Figure 8b
partly reflect handing the probe 9 numbers where eye hands it 3?

Primary metric per part: held-out multinomial log-loss gain
    gain = logloss(labels-only probe) - logloss(raw-z probe)
on the ordinary held-out images, 5-fold stratified by species. gain/K (K = block
width) is reported as a descriptive efficiency only. Secondary sensitivity: tail
probes on 40 random 3-of-9 coordinate subsets — "recoverability from three
randomly selected tail coordinates", never "total tail information".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diag_common as dc


def probe_cv(X: np.ndarray, y: np.ndarray, seed: int = dc.FOLD_SEED):
    """5-fold stratified held-out log-loss and accuracy of the 8c-family probe."""
    classes = np.unique(y)
    splitter = StratifiedKFold(n_splits=dc.N_FOLDS, shuffle=True, random_state=seed)
    losses, accs, weights = [], [], []
    for tr, te in splitter.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=1.0, random_state=dc.FOLD_SEED))
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])
        # align probability columns to the global class list (a fold can miss a class)
        full = np.full((len(te), len(classes)), 1e-9)
        col = {c: i for i, c in enumerate(model.classes_)}
        for c, i in col.items():
            full[:, np.searchsorted(classes, c)] = proba[:, i]
        full = full / full.sum(axis=1, keepdims=True)
        losses.append(log_loss(y[te], full, labels=classes))
        accs.append(accuracy_score(y[te], classes[full.argmax(axis=1)]))
        weights.append(len(te))
    w = np.asarray(weights, dtype=float)
    return (float(np.average(losses, weights=w)), float(np.average(accs, weights=w)))


def main():
    z, c, y, _, names = dc.load_eval()
    _, spans = dc.load_concepts()
    out = dc.out_dir()
    rng = np.random.default_rng(dc.SUBSET_SEED)

    n_species = len(np.unique(y))
    chance_logloss = float(np.log(n_species))
    rows = []
    for part in dc.ORDER:
        lo, hi = spans[part]
        K = hi - lo
        label_loss, label_acc = probe_cv(c[:, lo:hi].astype(float), y)
        raw_loss, raw_acc = probe_cv(z[:, lo:hi], y)
        gain = label_loss - raw_loss
        rows.append({"part": part, "K": K,
                     "chance_logloss": round(chance_logloss, 3),
                     "labels_logloss": round(label_loss, 3),
                     "raw_logloss": round(raw_loss, 3),
                     "labels_acc": round(label_acc, 3),
                     "raw_acc": round(raw_acc, 3),
                     "logloss_gain": round(gain, 3),
                     "gain_per_dimension_descriptive": round(gain / K, 4)})
    table = pd.DataFrame(rows)

    # Secondary sensitivity: tail restricted to 3 random coordinates, 40 repeats.
    lo, hi = spans["tail"]
    subset_losses = []
    for _ in range(40):
        cols = np.sort(rng.choice(np.arange(lo, hi), size=3, replace=False))
        loss, _acc = probe_cv(z[:, cols], y)
        subset_losses.append(loss)
    subset_losses = np.asarray(subset_losses)
    eye_row = table.loc[table.part == "eye"].iloc[0]
    subset_summary = pd.DataFrame([{
        "test": "tail on 3 random of 9 coordinates (40 subsets; sensitivity only)",
        "mean_logloss": round(float(subset_losses.mean()), 3),
        "min_logloss": round(float(subset_losses.min()), 3),
        "max_logloss": round(float(subset_losses.max()), 3),
        "eye_raw_logloss_K3_reference": eye_row.raw_logloss,
        "note": "subset repeats share data and are not independent uncertainty",
    }])

    table.to_csv(out / "d61_dimension_adjusted_information.csv", index=False)
    subset_summary.to_csv(out / "d61_tail_subset_sensitivity.csv", index=False)
    print("\nD6.1 · held-out species log-loss by input block (lower = more informative)")
    print(table.to_string(index=False))
    print("\nD6.1 · tail matched-dimension sensitivity (vs eye's 3 raw coordinates)")
    print(subset_summary.to_string(index=False))
    print("\nReading rule (predeclared): compare logloss_gain across parts for the "
          "fingerprint-beyond-labels claim; gain_per_dimension is descriptive only. "
          "If tail-on-3-coordinates still beats eye's raw block, tail's fingerprint "
          "is not merely a width artifact.")


if __name__ == "__main__":
    main()
