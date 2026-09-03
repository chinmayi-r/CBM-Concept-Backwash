"""D6.2 — Does a swapped part-score profile resemble donor, source, or neither?"""
from __future__ import annotations

import numpy as np
import pandas as pd

import diag_common as dc


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def transfer_scores(S, source_id, spans, z_eval, y_eval, eval_ids,
                    variant: str, stage: str, columns_by_part):
    """Absolute donor/source similarities plus their signed difference."""
    fold = dc.grouped_folds(S, source_id)
    records = []
    eval_id_arr = np.asarray(eval_ids, dtype=str)
    direct_overlap = set(eval_id_arr) & set(S[source_id].astype(str))
    species = np.unique(y_eval)
    expected_species = set(range(50))
    if set(species) != expected_species:
        raise RuntimeError(f"ordinary evaluation species are {sorted(species)}, expected 0..49")
    if not set(S.sid_src.astype(int)).issubset(expected_species) or not set(
            S.sid_donor.astype(int)).issubset(expected_species):
        raise RuntimeError("swap species indices are not aligned to ordinary evaluation 0..49")

    for part in dc.ORDER:
        lo, hi = spans[part]
        width = hi - lo
        d = S[S.part == part]
        d_fold = fold[d.index]
        X = d[columns_by_part[part]].to_numpy(dtype=float)
        for fold_id in range(dc.N_FOLDS):
            test_mask = d_fold == fold_id
            if not test_mask.any():
                continue
            held_out_ids = set(d.loc[test_mask, source_id].astype(str))
            # Direct equality is the only safe cross-table identity match. Zero
            # overlap is "unverified", not proof that the populations are disjoint.
            keep_eval = ~np.isin(eval_id_arr, list(held_out_ids & direct_overlap))
            ze = z_eval[keep_eval][:, lo:hi]
            ye = y_eval[keep_eval]
            mu = ze.mean(axis=0)
            sd = ze.std(axis=0)
            sd[sd == 0] = 1.0
            standardized_eval = (ze - mu) / sd
            signatures = np.full((50, width), np.nan)
            for sp in species:
                available = standardized_eval[ye == sp]
                if len(available):
                    signatures[sp] = available.mean(axis=0)
            missing_species = np.flatnonzero(np.isnan(signatures).any(axis=1)).tolist()
            if missing_species:
                raise RuntimeError(
                    f"{part} fold {fold_id} lacks ordinary signature data for "
                    f"species {missing_species}")

            rows = d[test_mask]
            block = (X[test_mask.to_numpy()] - mu) / sd
            donor_signature = signatures[rows.sid_donor.to_numpy(dtype=int)]
            source_signature = signatures[rows.sid_src.to_numpy(dtype=int)]
            if variant == "off_target":
                keep = np.ones((len(rows), width), dtype=bool)
                keep[np.arange(len(rows)), rows.var_src.to_numpy(dtype=int)] = False
                keep[np.arange(len(rows)), rows.var_donor.to_numpy(dtype=int)] = False
                block = np.where(keep, block, 0.0)
                donor_signature = np.where(keep, donor_signature, 0.0)
                source_signature = np.where(keep, source_signature, 0.0)

            donor_similarity = cosine(block, donor_signature)
            source_similarity = cosine(block, source_signature)
            for idx, donor_sim, source_sim in zip(
                    rows.index, donor_similarity, source_similarity):
                records.append({
                    "row_index": idx,
                    "part": part,
                    "fold": fold_id,
                    "stage": stage,
                    "variant": variant,
                    "outcome": S.loc[idx, "outcome"],
                    "original_image": str(S.loc[idx, source_id]),
                    "coordinates_used": width - 2 if variant == "off_target" else width,
                    "donor_similarity": float(donor_sim),
                    "source_similarity": float(source_sim),
                    "relative_transfer": float(donor_sim - source_sim),
                })
    return pd.DataFrame(records), len(direct_overlap)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    keys = ["stage", "variant", "part", "outcome"]
    for key, group in rows.groupby(keys):
        finite = group.dropna(subset=["donor_similarity", "source_similarity"])
        n_originals = finite.original_image.nunique()
        used = int(group.coordinates_used.iloc[0])
        interpretable = n_originals >= dc.MIN_ORIGINALS and used >= 2
        record = dict(zip(keys, key))
        record.update({
            "n_rows": len(group),
            "n_finite": len(finite),
            "n_originals": n_originals,
            "coordinates_used": used,
            "donor_similarity_median": finite.donor_similarity.median(),
            "source_similarity_median": finite.source_similarity.median(),
            "relative_transfer_median": finite.relative_transfer.median(),
            "relative_transfer_q25": finite.relative_transfer.quantile(.25),
            "relative_transfer_q75": finite.relative_transfer.quantile(.75),
            "interpretable": interpretable,
        })
        if len(finite) and n_originals:
            _, lo, hi = dc.clustered_metric_interval(
                finite.relative_transfer.to_numpy(),
                finite.original_image.to_numpy(), np.median)
            record["relative_transfer_ci_low"] = lo
            record["relative_transfer_ci_high"] = hi
        output.append(record)
    return pd.DataFrame(output)


def main():
    S, source_id, spans = dc.load_swaps()
    z_eval, _c, y_eval, eval_ids, _names = dc.load_eval()
    out = dc.out_dir()

    post_columns = {part: dc.block_columns(S, part) for part in dc.ORDER}
    pre_columns = {part: dc.pre_swap_block_columns(S, part) for part in dc.ORDER}
    has_any_pre = any(value is not None for value in pre_columns.values())
    has_all_pre = all(value is not None for value in pre_columns.values())
    if has_any_pre and not has_all_pre:
        missing = [part for part, value in pre_columns.items() if value is None]
        raise RuntimeError(f"pre-swap blocks are partial; missing {missing}")
    if has_all_pre:
        for part, columns in pre_columns.items():
            expected = spans[part][1] - spans[part][0]
            if len(columns) != expected:
                raise RuntimeError(
                    f"pre-swap block for {part} has {len(columns)} columns, expected {expected}")

    all_rows = []
    overlap_counts = set()
    stages = [("post", post_columns)]
    if has_all_pre:
        stages.insert(0, ("pre", pre_columns))
    for stage, columns in stages:
        for variant in ("full", "off_target"):
            rows, overlap = transfer_scores(
                S, source_id, spans, z_eval, y_eval, eval_ids,
                variant, stage, columns)
            all_rows.append(rows)
            overlap_counts.add(overlap)
    if len(overlap_counts) != 1:
        raise RuntimeError("profile-transfer overlap accounting changed between variants")
    overlap = overlap_counts.pop()
    if overlap:
        overlap_text = f"{overlap} exact identifier(s); held-out matches excluded per fold"
    else:
        overlap_text = ("0 exact identifier matches; ordinary predictions are treated as an "
                        "external reference population, not proven-disjoint images")
    print(f"[D6.2] ordinary/swap identity overlap: {overlap_text}")

    rows = pd.concat(all_rows, ignore_index=True)
    if has_all_pre:
        before = rows[rows.stage == "pre"][
            ["row_index", "variant", "relative_transfer"]].rename(
                columns={"relative_transfer": "relative_transfer_pre"})
        after = rows[rows.stage == "post"].merge(
            before, on=["row_index", "variant"], validate="one_to_one")
        after["relative_transfer_delta"] = (
            after.relative_transfer - after.relative_transfer_pre)
        after.to_csv(out / "d62_profile_transfer_pre_post.csv", index=False)
    else:
        print("[D6.2] no pre-swap per-part score blocks: post-swap similarities only")

    summary = summarize(rows)
    rows.to_csv(out / "d62_profile_transfer_rows.csv", index=False)
    summary.round(4).to_csv(out / "d62_profile_transfer_summary.csv", index=False)
    print("\nD6.2 · donor similarity and source similarity are printed separately")
    print(summary.round(3).to_string(index=False))
    print("\nReading rule: relative_transfer = donor similarity - source similarity. "
          "Positive is relatively donor-like and negative relatively source-like. "
          "The two absolute similarity columns distinguish 'similar to both' from "
          "'similar to neither'; the difference alone cannot. Eye off-target uses one "
          "coordinate and is marked non-interpretable. Intervals resample original "
          "images and are not training-seed uncertainty.")


if __name__ == "__main__":
    main()
