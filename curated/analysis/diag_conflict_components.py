"""D6.3 — Conflict→response components (predeclared, DECISIONS.md D6.3).

Question: do training labels that contradict visibility map onto the specific
score movements they could damage? Predeclared mappings:
    donor-value conflict_rate  -> donor_gain
    source-value conflict_rate -> source_decrease
    both conflict rates        -> m_cf
Levels: (a) exact-value aggregation (<=26 points; EXPLORATORY), (b) row-level
OLS with controls (log1p pixel_count_cf, donor support, part indicators) and
cluster-robust standard errors by original source image.

Descriptive bridge only: the causal label test is notebook 02rl. This predicts
WHERE relabeling should help (higher-conflict values); 02rl's paired replay is
the test of whether it does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import diag_common as dc


def cluster_robust_ols(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, names):
    """OLS with CR1 cluster-robust (sandwich) standard errors.

    Returns a DataFrame of coefficient, clustered SE, and t statistic. The
    cluster count (not the row count) carries the effective sample size.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    labels, inv = np.unique(clusters, return_inverse=True)
    G = len(labels)
    meat = np.zeros((k, k))
    for g in range(G):
        Xg = X[inv == g]
        ug = resid[inv == g]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    cov = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({"term": names, "coef": beta, "cluster_se": se,
                         "t": beta / se}), G


def standardize(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        sd = out[c].std()
        out[c] = (out[c] - out[c].mean()) / (sd if sd else 1.0)
    return out


def main():
    S, source_id, spans = dc.load_swaps()
    conflict = dc.load_conflict_rates()
    out = dc.out_dir()

    n_undefined = int(conflict.conflict_rate.isna().sum())
    if n_undefined:
        print(f"[D6.3] {n_undefined} concept(s) have zero positive labels; their "
              f"conflict rate is defined as 0.0 (no positives -> no conflict)")
    key = conflict.set_index(["part", "value"]).conflict_rate.fillna(0.0)
    S = S.copy()
    S["donor_conflict"] = [key.get((p, int(v))) for p, v in zip(S.part, S.var_donor)]
    S["source_conflict"] = [key.get((p, int(v))) for p, v in zip(S.part, S.var_src)]
    if S.donor_conflict.isna().any() or S.source_conflict.isna().any():
        raise RuntimeError("some swap values have no conflict-rate row — check value indexing")
    support = (S.groupby(["part", "var_donor"]).sid_donor.nunique()
               .rename("donor_support"))
    S = S.join(support, on=["part", "var_donor"])
    S["log_pixels"] = np.log1p(S.pixel_count_cf)

    # (a) exact-value level — EXPLORATORY (<=26 points, support/visibility not held fixed)
    donor_level = (S.groupby(["part", "var_donor"])
                   .agg(donor_conflict=("donor_conflict", "first"),
                        mean_donor_gain=("donor_gain", "mean"),
                        n_rows=("donor_gain", "size")).reset_index())
    source_level = (S.groupby(["part", "var_src"])
                    .agg(source_conflict=("source_conflict", "first"),
                         mean_source_decrease=("source_decrease", "mean"),
                         mean_m_orig=("m_orig", "mean"),
                         n_rows=("source_decrease", "size")).reset_index())
    value_rows = []
    for part, g in donor_level.groupby("part"):
        value_rows.append({"part": part, "mapping": "donor_conflict -> donor_gain",
                           "n_values": len(g),
                           "spearman": g.donor_conflict.corr(g.mean_donor_gain, method="spearman")})
    for part, g in source_level.groupby("part"):
        value_rows.append({"part": part, "mapping": "source_conflict -> source_decrease",
                           "n_values": len(g),
                           "spearman": g.source_conflict.corr(g.mean_source_decrease, method="spearman")})
        value_rows.append({"part": part, "mapping": "source_conflict -> m_orig",
                           "n_values": len(g),
                           "spearman": g.source_conflict.corr(g.mean_m_orig, method="spearman")})
    value_table = pd.DataFrame(value_rows)

    # (b) row-level OLS with controls and cluster-robust SEs by original image
    part_dummies = pd.get_dummies(S.part, prefix="part", drop_first=True).astype(float)
    controls = ["log_pixels", "donor_support"]
    clusters = S[source_id].astype(str).to_numpy()

    def run(target: str, conflict_col: str):
        cols = [conflict_col] + controls
        Z = standardize(S[cols + [target]], cols)
        X = np.column_stack([np.ones(len(S)),
                             Z[cols].to_numpy(),
                             part_dummies.to_numpy()])
        names = ["intercept"] + cols + list(part_dummies.columns)
        table, G = cluster_robust_ols(X, Z[target].to_numpy(dtype=float), clusters, names)
        table.insert(0, "target", target)
        table.insert(1, "clusters", G)
        return table

    row_tables = pd.concat([
        run("donor_gain", "donor_conflict"),
        run("source_decrease", "source_conflict"),
        run("m_cf", "donor_conflict"),
        run("m_cf", "source_conflict"),
        run("m_orig", "source_conflict"),
    ], ignore_index=True).round(4)

    value_table.round(3).to_csv(out / "d63_conflict_value_level.csv", index=False)
    row_tables.to_csv(out / "d63_conflict_row_level_ols.csv", index=False)

    print("\nD6.3(a) · exact-value level (EXPLORATORY: few values per part; "
          "support/visibility not held fixed here)")
    print(value_table.round(3).to_string(index=False))
    print("\nD6.3(b) · row-level OLS, standardized predictors, controls "
          "(log pixels, donor support, part indicators), cluster-robust SE by "
          "original image (~%d clusters)" % row_tables.clusters.iloc[0])
    focus = row_tables[row_tables.term.isin(["donor_conflict", "source_conflict"])]
    print(focus.to_string(index=False))
    print("\nFull coefficient tables: d63_conflict_row_level_ols.csv")
    print("\nReading rule (predeclared): the supervision story gains support only "
          "if the matched mappings hold with controls in place (donor conflict "
          "depressing donor_gain; source conflict depressing source_decrease). "
          "A generic 'tail has conflict and tail is bad' pattern does not count. "
          "Causal credit is decided by notebook 02rl's paired replay, not here.")


if __name__ == "__main__":
    main()
