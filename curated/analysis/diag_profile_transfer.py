"""D6.2 — Part-profile transfer, full-block and off-target (predeclared, DECISIONS.md D6.2).

Question: after a controlled swap, does the replaced part's WHOLE score pattern
resemble the donor species' typical pattern, the source's, or neither? The
off-target variant deletes the removed (var_src) and inserted (var_donor)
coordinates first, so it cannot restate the already-measured donor/source margin.

Score per swap (predeclared): cosine(post-swap block, donor signature)
  - cosine(post-swap block, source signature); positive = donor-like.
Signatures = per-species mean of the part block over ordinary held-out images,
computed inside each training fold, coordinates standardized by training-fold
std. Any evaluation image identified as a held-out-fold original is excluded
from that fold's signatures (overlap measured and printed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import diag_common as dc


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between matrix a [n,k] and vector-per-row b [n,k]."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / den, np.nan)


def transfer_scores(S, source_id, spans, z_eval, y_eval, eval_ids, variant: str):
    """Per-swap transfer score for one variant ('full' or 'off_target')."""
    fold = dc.grouped_folds(S, source_id)
    records = []
    eval_id_arr = np.asarray(eval_ids, dtype=str)
    swap_orig_ids = set(S[source_id].astype(str))
    overlap = sorted(set(eval_id_arr) & swap_orig_ids)
    for part in dc.ORDER:
        lo, hi = spans[part]
        K = hi - lo
        cols = dc.block_columns(S, part)
        d = S[S.part == part]
        d_fold = fold[d.index]
        X = d[cols].to_numpy(dtype=float)
        for f in range(dc.N_FOLDS):
            te = d_fold == f
            if not te.any():
                continue
            # training-fold original ids define which eval images to exclude
            held_out_ids = set(d.loc[te, source_id].astype(str))
            keep_eval = ~np.isin(eval_id_arr, list(held_out_ids))
            ze = z_eval[keep_eval][:, lo:hi]
            ye = y_eval[keep_eval]
            mu = ze.mean(axis=0)
            sd = ze.std(axis=0)
            sd[sd == 0] = 1.0
            zs = (ze - mu) / sd
            signatures = np.full((int(y_eval.max()) + 1, K), np.nan)
            for sp in np.unique(ye):
                signatures[sp] = zs[ye == sp].mean(axis=0)
            Xte = (X[te.to_numpy()] - mu) / sd
            rows = d[te]
            sig_donor = signatures[rows.sid_donor.to_numpy(dtype=int)]
            sig_source = signatures[rows.sid_src.to_numpy(dtype=int)]
            if variant == "off_target":
                keep_cols = np.ones((len(rows), K), dtype=bool)
                keep_cols[np.arange(len(rows)), rows.var_src.to_numpy(dtype=int)] = False
                keep_cols[np.arange(len(rows)), rows.var_donor.to_numpy(dtype=int)] = False
                # zero out dropped coordinates so cosine ignores them
                Xv = np.where(keep_cols, Xte, 0.0)
                sd_v = np.where(keep_cols, sig_donor, 0.0)
                ss_v = np.where(keep_cols, sig_source, 0.0)
            else:
                Xv, sd_v, ss_v = Xte, sig_donor, sig_source
            score = cosine(Xv, sd_v) - cosine(Xv, ss_v)
            for idx, sc in zip(rows.index, score):
                records.append({
                    "row_index": idx, "part": part, "fold": f, "variant": variant,
                    "outcome": S.loc[idx, "outcome"],
                    "original_image": str(S.loc[idx, source_id]),
                    "off_target_coords": K - 2 if variant == "off_target" else K,
                    "transfer_score": float(sc),
                })
    return pd.DataFrame(records), overlap


def summarize(R: pd.DataFrame) -> pd.DataFrame:
    g = (R.groupby(["variant", "part", "outcome"])
         .agg(n_rows=("transfer_score", "size"),
              n_originals=("original_image", "nunique"),
              median=("transfer_score", "median"),
              q25=("transfer_score", lambda s: s.quantile(.25)),
              q75=("transfer_score", lambda s: s.quantile(.75)))
         .reset_index())
    g["interpretable"] = g.n_originals >= dc.MIN_ORIGINALS
    return g.round(3)


def main():
    S, source_id, spans = dc.load_swaps()
    z_eval, _c, y_eval, eval_ids, _names = dc.load_eval()
    out = dc.out_dir()

    all_rows = []
    for variant in ["full", "off_target"]:
        R, overlap = transfer_scores(S, source_id, spans, z_eval, y_eval, eval_ids, variant)
        all_rows.append(R)
    R = pd.concat(all_rows, ignore_index=True)
    print(f"[D6.2] eval/swap-original id overlap: {len(overlap)} image(s)"
          + (" — excluded per fold" if overlap else " (populations disjoint)"))
    n_nan = int(R.transfer_score.isna().sum())
    if n_nan:
        print(f"[D6.2] WARNING: {n_nan} rows have no score (a species had no "
              f"signature images in some training fold); they are excluded from medians.")

    pre_cols = dc.pre_swap_block_columns(S, "tail")
    if pre_cols is None:
        print("[D6.2] pre-swap per-part blocks not present in the CSV: "
              "reporting post-swap transfer only (as predeclared).")

    summary = summarize(R)
    R.to_csv(out / "d62_profile_transfer_rows.csv", index=False)
    summary.to_csv(out / "d62_profile_transfer_summary.csv", index=False)

    print("\nD6.2 · transfer score = cos(block, donor signature) - cos(block, source "
          "signature); positive = donor-like")
    for variant in ["full", "off_target"]:
        print(f"\n--- variant: {variant}"
              + ("  (removed+inserted coordinates deleted; eye keeps 1 coordinate — flagged)"
                 if variant == "off_target" else ""))
        v = summary[summary.variant == variant].drop(columns="variant")
        print(v.to_string(index=False))

    print("\nReading rule (predeclared): the retention story is supported for a part "
          "only if the OFF-TARGET score is materially more source-like (lower) in "
          "'donorward, source wins' rows than in 'donor wins' rows, among "
          "interpretable groups. Full-block results contextualize but cannot "
          "support retention on their own.")


if __name__ == "__main__":
    main()
