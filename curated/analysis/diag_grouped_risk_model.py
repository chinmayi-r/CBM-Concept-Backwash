"""D6.4 — Grouped continuous risk model (predeclared, DECISIONS.md D6.4).

Successor question to Figure 9: do the measured pre-outcome factors predict the
final margin for UNSEEN original images, when encoded continuously instead of
as sparse categorical lookups? Primary target: m_cf. Secondary: the controlled
event indicator. Predictors are all available before the outcome; no post-swap
quantities.

Variants: (a) generic + part indicators (predictive ceiling);
          (b) generic only (transport candidate — no part/species identity).
Nested CV: outer = the Figure 8c grouped 5-fold scheme; inner = grouped 4-fold
on training folds for alpha in {0.01, 0.1, 1, 10, 100}.
Baselines: global mean, part-only mean, m_orig-only ridge.
Transport (variant b only): leave-one-exact-donor-value-out and
leave-one-part-out (5 parts — unstable by construction, reported as such).
Coefficients are predictive associations, never causal contributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

import diag_common as dc

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
GENERIC = ["m_orig", "log_pixels", "donor_support",
           "donor_conflict", "source_conflict", "alternatives_in_part"]


def build_features(S: pd.DataFrame, spans) -> pd.DataFrame:
    conflict = dc.load_conflict_rates()
    key = conflict.set_index(["part", "value"]).conflict_rate.fillna(0.0)
    F = S.copy()
    F["donor_conflict"] = [key.get((p, int(v))) for p, v in zip(F.part, F.var_donor)]
    F["source_conflict"] = [key.get((p, int(v))) for p, v in zip(F.part, F.var_src)]
    support = F.groupby(["part", "var_donor"]).sid_donor.nunique().rename("donor_support")
    F = F.join(support, on=["part", "var_donor"])
    F["log_pixels"] = np.log1p(F.pixel_count_cf)
    F["alternatives_in_part"] = F.part.map({p: hi - lo for p, (lo, hi) in spans.items()})
    F["event"] = ((F.response_delta > 0) & (F.m_cf <= 0)).astype(int)
    if F[GENERIC].isna().any().any():
        raise RuntimeError("missing feature values — check conflict/support joins")
    return F


def matrix(F: pd.DataFrame, with_part: bool):
    X = F[GENERIC].to_numpy(dtype=float)
    names = list(GENERIC)
    if with_part:
        D = pd.get_dummies(F.part, prefix="part", drop_first=True).astype(float)
        X = np.column_stack([X, D.to_numpy()])
        names += list(D.columns)
    return X, names


def fit_ridge(Xtr, ytr, Xte, groups_tr, alphas=ALPHAS):
    """Ridge with inner grouped 4-fold alpha selection; features standardized on train."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    units = np.unique(groups_tr)
    rng = np.random.default_rng(dc.FOLD_SEED)
    unit_fold = dict(zip(units, rng.integers(0, 4, size=len(units))))
    inner = np.vectorize(unit_fold.get)(groups_tr)
    best_alpha, best_err = None, np.inf
    for a in alphas:
        errs = []
        for f in range(4):
            m = Ridge(alpha=a).fit(Xtr[inner != f], ytr[inner != f])
            errs.append(np.mean((ytr[inner == f] - m.predict(Xtr[inner == f])) ** 2))
        err = float(np.mean(errs))
        if err < best_err:
            best_alpha, best_err = a, err
    model = Ridge(alpha=best_alpha).fit(Xtr, ytr)
    return model.predict(Xte), best_alpha, model.coef_


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def outer_cv(F, X, y, folds, groups):
    preds = np.full(len(F), np.nan)
    alphas_used, coefs = [], []
    for f in range(dc.N_FOLDS):
        tr, te = folds != f, folds == f
        p, a, coef = fit_ridge(X[tr], y[tr], X[te], groups[tr])
        preds[te] = p
        alphas_used.append(a)
        coefs.append(coef)
    return preds, alphas_used, np.mean(coefs, axis=0)


def main():
    S, source_id, spans = dc.load_swaps()
    F = build_features(S, spans)
    folds = dc.grouped_folds(F, source_id).to_numpy()
    groups = F[source_id].astype(str).to_numpy()
    y = F.m_cf.to_numpy(dtype=float)
    out = dc.out_dir()

    results, coef_rows = [], []

    # Baselines under the same grouped outer folds
    for name, predictor in [
        ("baseline: global mean", lambda tr, te: np.full(te.sum(), y[tr].mean())),
        ("baseline: part-only mean", lambda tr, te: F[te].part.map(
            F[tr].groupby("part").m_cf.mean()).to_numpy()),
    ]:
        preds = np.full(len(F), np.nan)
        for f in range(dc.N_FOLDS):
            tr, te = folds != f, folds == f
            preds[te] = predictor(tr, te)
        results.append({"model": name, "RMSE": rmse(y, preds), "MAE": mae(y, preds)})

    X_morig = F[["m_orig"]].to_numpy(dtype=float)
    preds, _, _ = outer_cv(F, X_morig, y, folds, groups)
    results.append({"model": "baseline: m_orig-only ridge",
                    "RMSE": rmse(y, preds), "MAE": mae(y, preds)})

    for with_part, label in [(True, "ridge: generic + part indicators"),
                             (False, "ridge: generic only (transport candidate)")]:
        X, names = matrix(F, with_part)
        preds, alphas_used, coef = outer_cv(F, X, y, folds, groups)
        results.append({"model": label, "RMSE": rmse(y, preds), "MAE": mae(y, preds),
                        "alphas": str(sorted(set(alphas_used)))})
        coef_rows += [{"model": label, "term": n,
                       "mean_coef_predictive_association": round(c, 4)}
                      for n, c in zip(names, coef)]

    # Secondary target: controlled event (logistic ridge, same folds)
    X, _ = matrix(F, True)
    ev = F.event.to_numpy()
    ev_pred = np.full(len(F), np.nan)
    for f in range(dc.N_FOLDS):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        m = LogisticRegression(max_iter=4000, C=1.0, random_state=dc.FOLD_SEED)
        m.fit((X[tr] - mu) / sd, ev[tr])
        ev_pred[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    event_auc = roc_auc_score(ev, ev_pred)

    # Transport tests: generic features only
    Xg, _ = matrix(F, False)
    transport = []
    for part, g in F.groupby("part"):
        for val, gg in g.groupby("var_donor"):
            te = ((F.part == part) & (F.var_donor == val)).to_numpy()
            p, _, _ = fit_ridge(Xg[~te], y[~te], Xg[te], groups[~te])
            transport.append({"test": "leave-one-donor-value-out",
                              "held_out": f"{part}_{int(val)}",
                              "n": int(te.sum()), "RMSE": rmse(y[te], p)})
    for part in dc.ORDER:
        te = (F.part == part).to_numpy()
        p, _, _ = fit_ridge(Xg[~te], y[~te], Xg[te], groups[~te])
        transport.append({"test": "leave-one-part-out (unstable: 5 parts)",
                          "held_out": part, "n": int(te.sum()),
                          "RMSE": rmse(y[te], p)})
    T = pd.DataFrame(transport).round(3)

    R = pd.DataFrame(results).round(3)
    C = pd.DataFrame(coef_rows)
    R.to_csv(out / "d64_risk_model_heldout.csv", index=False)
    C.to_csv(out / "d64_risk_model_coefficients.csv", index=False)
    T.to_csv(out / "d64_risk_model_transport.csv", index=False)

    print("\nD6.4 · held-out prediction of final margin m_cf "
          "(grouped 5-fold by original image; Figure 9 lookup RMSE reference: "
          "part-only 3.333, +visibility 3.098)")
    print(R.to_string(index=False))
    print(f"\nD6.4 · secondary target — controlled event, grouped held-out AUC: "
          f"{event_auc:.3f}")
    print("\nD6.4 · transport tests (generic features, no part/species identity)")
    lovo = T[T.test == "leave-one-donor-value-out"]
    print(f"leave-one-donor-value-out: median RMSE {lovo.RMSE.median():.3f} "
          f"(range {lovo.RMSE.min():.3f}-{lovo.RMSE.max():.3f}, {len(lovo)} values)")
    print(T[T.test != "leave-one-donor-value-out"].to_string(index=False))
    print("\nCoefficients (predictive associations, not causal contributions): "
          "d64_risk_model_coefficients.csv")
    print("\nReading rule (predeclared): image-grouped results claim generalization "
          "to new FunnyBird images only. Only the value/part-held-out rows may "
          "support any 'new concepts' language, and leave-one-part-out is "
          "unstable by construction with five parts.")


if __name__ == "__main__":
    main()
