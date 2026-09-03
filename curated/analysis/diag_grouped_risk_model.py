"""D6.4 — Can measured pre-score factors predict the final swap margin?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import diag_common as dc

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
CORE = ["m_orig", "log_pixels", "donor_support",
        "donor_conflict", "source_conflict"]
STRUCTURAL = CORE + ["alternatives_in_part"]


def build_features(S: pd.DataFrame, spans) -> pd.DataFrame:
    conflict = dc.load_conflict_rates()
    key = conflict.set_index(["part", "value"]).conflict_rate
    F = S.copy()
    F["donor_conflict"] = [key.get((p, int(v))) for p, v in zip(F.part, F.var_donor)]
    F["source_conflict"] = [key.get((p, int(v))) for p, v in zip(F.part, F.var_src)]
    if F[["donor_conflict", "source_conflict"]].isna().any().any():
        raise RuntimeError("a used swap value has undefined conflict rate; it cannot be coded as zero")
    # Support is fixed dataset structure (number of species represented for an
    # exact value), computed without using any model score or outcome.
    donor_support = F.groupby(["part", "var_donor"]).sid_donor.nunique().rename(
        "donor_support")
    F = F.join(donor_support, on=["part", "var_donor"])
    F["log_pixels"] = np.log1p(F.pixel_count_cf)
    F["alternatives_in_part"] = F.part.map(
        {part: hi - lo for part, (lo, hi) in spans.items()})
    F["event"] = dc.controlled_event(F).astype(int)
    if F[STRUCTURAL].isna().any().any():
        raise RuntimeError("missing risk-model feature values")
    return F


def matrix(F: pd.DataFrame, feature_names, with_part=False):
    X = F[list(feature_names)].to_numpy(dtype=float)
    names = list(feature_names)
    if with_part:
        dummies = pd.get_dummies(F.part, prefix="part", drop_first=True).astype(float)
        X = np.column_stack([X, dummies.to_numpy()])
        names += list(dummies.columns)
    return X, names


def _scale(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def _inner_splits(groups):
    unique = np.unique(groups)
    folds = min(4, len(unique))
    if folds < 2:
        raise RuntimeError("nested validation needs at least two original-image groups")
    return GroupKFold(folds).split(np.zeros(len(groups)), groups=groups)


def fit_ridge(X_train, y_train, X_test, groups_train):
    """Choose alpha with genuinely nested group CV, including fold-local scaling."""
    errors = {alpha: [] for alpha in ALPHAS}
    for inner_train, inner_valid in _inner_splits(groups_train):
        x_train, x_valid = _scale(X_train[inner_train], X_train[inner_valid])
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, y_train[inner_train])
            squared = (y_train[inner_valid] - model.predict(x_valid)) ** 2
            errors[alpha].extend(squared.tolist())
    best = min(ALPHAS, key=lambda alpha: np.mean(errors[alpha]))
    x_train, x_test = _scale(X_train, X_test)
    model = Ridge(alpha=best).fit(x_train, y_train)
    return model.predict(x_test), best, model.coef_


def fit_logistic(X_train, y_train, X_test, groups_train):
    """Choose logistic regularization inside grouped, fold-locally scaled CV."""
    losses = {alpha: [] for alpha in ALPHAS}
    for inner_train, inner_valid in _inner_splits(groups_train):
        if len(np.unique(y_train[inner_train])) < 2:
            raise RuntimeError("an inner event-training fold contains one class")
        x_train, x_valid = _scale(X_train[inner_train], X_train[inner_valid])
        for alpha in ALPHAS:
            model = LogisticRegression(
                max_iter=4000, C=1.0 / alpha, random_state=dc.FOLD_SEED)
            model.fit(x_train, y_train[inner_train])
            probability = model.predict_proba(x_valid)[:, 1]
            losses[alpha].append(log_loss(
                y_train[inner_valid], probability, labels=[0, 1]))
    best = min(ALPHAS, key=lambda alpha: np.mean(losses[alpha]))
    x_train, x_test = _scale(X_train, X_test)
    model = LogisticRegression(
        max_iter=4000, C=1.0 / best, random_state=dc.FOLD_SEED)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1], best


def outer_ridge(X, y, folds, groups):
    predictions = np.full(len(y), np.nan)
    alphas, coefficients = [], []
    for fold in range(dc.N_FOLDS):
        train, test = folds != fold, folds == fold
        prediction, alpha, coefficient = fit_ridge(
            X[train], y[train], X[test], groups[train])
        predictions[test] = prediction
        alphas.append(alpha)
        coefficients.append(coefficient)
    if np.isnan(predictions).any():
        raise RuntimeError("ridge outer-fold predictions are incomplete")
    return predictions, alphas, np.mean(coefficients, axis=0)


def outer_logistic(X, y, folds, groups):
    predictions = np.full(len(y), np.nan)
    alphas = []
    for fold in range(dc.N_FOLDS):
        train, test = folds != fold, folds == fold
        prediction, alpha = fit_logistic(X[train], y[train], X[test], groups[train])
        predictions[test] = prediction
        alphas.append(alpha)
    if np.isnan(predictions).any():
        raise RuntimeError("logistic outer-fold predictions are incomplete")
    return predictions, alphas


def _error_row(name, y, prediction, groups, alphas=None):
    squared = (y - prediction) ** 2
    absolute = np.abs(y - prediction)
    mse, mse_lo, mse_hi = dc.clustered_metric_interval(squared, groups, np.mean)
    mae, mae_lo, mae_hi = dc.clustered_metric_interval(absolute, groups, np.mean)
    row = {
        "model": name,
        "RMSE": np.sqrt(mse),
        "RMSE_ci_low": np.sqrt(max(0.0, mse_lo)),
        "RMSE_ci_high": np.sqrt(max(0.0, mse_hi)),
        "MAE": mae,
        "MAE_ci_low": mae_lo,
        "MAE_ci_high": mae_hi,
    }
    if alphas is not None:
        row["alphas"] = str(sorted(set(alphas)))
    return row


def main():
    S, source_id, spans = dc.load_swaps()
    F = build_features(S, spans)
    folds = dc.grouped_folds(F, source_id).to_numpy()
    groups = F[source_id].astype(str).to_numpy()
    y = F.m_cf.to_numpy(dtype=float)
    out = dc.out_dir()

    results, coefficient_rows = [], []
    for name, predictor in [
        ("baseline: global mean", lambda tr, te: np.full(te.sum(), y[tr].mean())),
        ("baseline: part-only mean", lambda tr, te: F.loc[te, "part"].map(
            F.loc[tr].groupby("part").m_cf.mean()).to_numpy()),
    ]:
        prediction = np.full(len(F), np.nan)
        for fold in range(dc.N_FOLDS):
            train, test = folds != fold, folds == fold
            prediction[test] = predictor(train, test)
        results.append(_error_row(name, y, prediction, groups))

    specifications = [
        (CORE, False, "ridge: continuous factors (no explicit part/alternatives)"),
        (STRUCTURAL, False, "ridge: + alternatives count (part-structure proxy)"),
        (STRUCTURAL, True, "ridge: + alternatives count + explicit part indicators"),
    ]
    X_m_orig, _ = matrix(F, ["m_orig"])
    prediction, alphas, _ = outer_ridge(X_m_orig, y, folds, groups)
    results.append(_error_row("baseline: m_orig-only ridge", y, prediction, groups, alphas))
    for features, with_part, label in specifications:
        X, names = matrix(F, features, with_part)
        prediction, alphas, coefficients = outer_ridge(X, y, folds, groups)
        results.append(_error_row(label, y, prediction, groups, alphas))
        coefficient_rows.extend({
            "model": label,
            "term": name,
            "mean_standardized_fold_coefficient": coefficient,
        } for name, coefficient in zip(names, coefficients))

    # Secondary binary event: include comparable baselines and tune regularization.
    event = F.event.to_numpy(dtype=int)
    event_rows = []
    for label, use_part in [
        ("event baseline: global rate", False),
        ("event baseline: part-only rate", True),
    ]:
        probability = np.full(len(F), np.nan)
        for fold in range(dc.N_FOLDS):
            train, test = folds != fold, folds == fold
            if use_part:
                rates = F.loc[train].groupby("part").event.mean()
                probability[test] = F.loc[test, "part"].map(rates).to_numpy()
            else:
                probability[test] = event[train].mean()
        event_rows.append({
            "model": label,
            "AUC": roc_auc_score(event, probability),
            "Brier": np.mean((event - probability) ** 2),
            "alphas": "not applicable",
        })
    for features, with_part, label in [
        (["m_orig"], False, "event baseline: m_orig only"),
        (CORE, False, "event: continuous factors (no explicit part/alternatives)"),
        (STRUCTURAL, True, "event: factors + alternatives + part indicators"),
    ]:
        X, _ = matrix(F, features, with_part)
        probability, alphas = outer_logistic(X, event, folds, groups)
        event_rows.append({
            "model": label,
            "AUC": roc_auc_score(event, probability),
            "Brier": np.mean((event - probability) ** 2),
            "alphas": str(sorted(set(alphas))),
        })

    # Transport uses the strict set: no part indicator and no alternatives count.
    X_transport, _ = matrix(F, CORE)
    transport = []
    for part, part_rows in F.groupby("part"):
        for value in sorted(part_rows.var_donor.unique()):
            test = ((F.part == part) & (F.var_donor == value)).to_numpy()
            prediction, alpha, _ = fit_ridge(
                X_transport[~test], y[~test], X_transport[test], groups[~test])
            transport.append({
                "test": "leave-one-exact-donor-value-out",
                "held_out": f"{part}_{int(value)}",
                "n_rows": int(test.sum()),
                "n_originals": F.loc[test, source_id].nunique(),
                "RMSE": np.sqrt(np.mean((y[test] - prediction) ** 2)),
                "alpha": alpha,
            })
    for part in dc.ORDER:
        test = (F.part == part).to_numpy()
        prediction, alpha, _ = fit_ridge(
            X_transport[~test], y[~test], X_transport[test], groups[~test])
        transport.append({
            "test": "leave-one-part-out (five-part stress test)",
            "held_out": part,
            "n_rows": int(test.sum()),
            "n_originals": F.loc[test, source_id].nunique(),
            "RMSE": np.sqrt(np.mean((y[test] - prediction) ** 2)),
            "alpha": alpha,
        })

    result_table = pd.DataFrame(results)
    event_table = pd.DataFrame(event_rows)
    transport_table = pd.DataFrame(transport)
    result_table.round(4).to_csv(out / "d64_risk_model_heldout.csv", index=False)
    event_table.round(4).to_csv(out / "d64_event_model_heldout.csv", index=False)
    pd.DataFrame(coefficient_rows).round(5).to_csv(
        out / "d64_risk_model_coefficients.csv", index=False)
    transport_table.round(4).to_csv(out / "d64_risk_model_transport.csv", index=False)

    print("\nD6.4 · grouped held-out prediction of final raw-logit margin m_cf")
    print(result_table.round(3).to_string(index=False))
    print("\nD6.4 · secondary controlled-event prediction")
    print(event_table.round(3).to_string(index=False))
    print("\nD6.4 · transport without explicit part or alternatives-count features")
    value_rows = transport_table[
        transport_table.test == "leave-one-exact-donor-value-out"]
    print(f"exact-value median RMSE={value_rows.RMSE.median():.3f}; "
          f"range={value_rows.RMSE.min():.3f}-{value_rows.RMSE.max():.3f}")
    print(transport_table[transport_table.test.str.startswith("leave-one-part")]
          .round(3).to_string(index=False))
    print("\nReading rule: the counterfactual mask area is measured after rendering "
          "the swap but before scoring it; it is not a model-output predictor. "
          "Alternatives count is explicitly labelled a part-structure proxy and is "
          "excluded from strict transport. Other feature distributions can still "
          "correlate with part, so this is not guaranteed part-invariant. Support is "
          "a supplied dataset descriptor, not learned from the model outcome. "
          "Image-held-out results address new images; "
          "exact-value-held-out results address a new value within a familiar part; "
          "the five-part holdout is only a stress test. Intervals resample original "
          "images and do not represent training-seed uncertainty.")


if __name__ == "__main__":
    main()
