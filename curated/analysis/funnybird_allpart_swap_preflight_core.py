"""Pure calculations for the FunnyBird all-part controlled-swap preflight."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd


def balanced_swap_rows(frame: pd.DataFrame, parts: list[str], per_part: int) -> pd.DataFrame:
    """Select deterministic round-robin rows across every donor value of each part."""
    required = {
        "part", "var_src", "var_donor", "orig_render_id", "render_id",
        "image_cf_sha256",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"swap rows lack selection columns: {sorted(missing)}")
    if per_part <= 0:
        raise ValueError("per_part must be positive")

    selected: list[pd.Series] = []
    for part in parts:
        group = frame.loc[frame.part == part].copy()
        if group.empty:
            raise ValueError(f"no controlled swaps for input part {part}")
        donor_values = sorted(group.var_donor.astype(int).unique().tolist())
        if per_part < len(donor_values):
            raise ValueError(
                f"{part}: rows-per-part={per_part} cannot cover all "
                f"{len(donor_values)} donor values")
        buckets: OrderedDict[int, pd.DataFrame] = OrderedDict()
        for value in donor_values:
            bucket = group.loc[group.var_donor.astype(int) == value].sort_values(
                ["orig_render_id", "var_src", "render_id"], kind="stable")
            buckets[value] = bucket.reset_index(drop=True)

        used_hashes: set[str] = set()
        depth = 0
        part_rows: list[pd.Series] = []
        while len(part_rows) < per_part:
            added = False
            for value, bucket in buckets.items():
                while depth < len(bucket):
                    row = bucket.iloc[depth]
                    key = str(row.image_cf_sha256)
                    if key not in used_hashes:
                        part_rows.append(row)
                        used_hashes.add(key)
                        added = True
                        break
                    # A duplicate RGB can occur for equivalent renderer requests.
                    # Search forward only inside this bucket.
                    found = None
                    for position in range(depth + 1, len(bucket)):
                        candidate = bucket.iloc[position]
                        if str(candidate.image_cf_sha256) not in used_hashes:
                            found = candidate
                            break
                    if found is not None:
                        part_rows.append(found)
                        used_hashes.add(str(found.image_cf_sha256))
                        added = True
                    break
                if len(part_rows) == per_part:
                    break
            if not added:
                break
            depth += 1
        if len(part_rows) != per_part:
            raise ValueError(
                f"{part}: found only {len(part_rows)} distinct counterfactual RGBs; "
                f"requested {per_part}")
        selected.extend(part_rows)
    return pd.DataFrame(selected).reset_index(drop=True)


def best_other_margin(values: np.ndarray, true_local: int) -> float:
    """Raw score of the true value minus the largest competing value."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("concept block must be a one-dimensional vector of width >= 2")
    if not 0 <= true_local < len(values):
        raise ValueError(f"true local index {true_local} outside width {len(values)}")
    others = np.delete(values, true_local)
    return float(values[true_local] - np.max(others))


def decompose_swap(
    *,
    z_orig: np.ndarray,
    z_cf: np.ndarray,
    class_weights: np.ndarray,
    source_species: int,
    donor_species: int,
    input_part: str,
    source_value: int,
    donor_value: int,
    original_values: dict[str, int],
    spans: dict[str, tuple[int, int]],
) -> list[dict]:
    """Decompose one controlled swap across every output concept block.

    ``class_gap_shift`` is the contribution of one block to the change in the
    saved source-minus-donor species logit gap. Positive favours the source;
    negative favours the donor. For unchanged blocks, ``unchanged_margin_change``
    measures whether the original value becomes easier or harder to distinguish
    from its strongest within-block competitor.
    """
    z_orig = np.asarray(z_orig, dtype=float)
    z_cf = np.asarray(z_cf, dtype=float)
    weights = np.asarray(class_weights, dtype=float)
    if z_orig.shape != z_cf.shape or z_orig.ndim != 1:
        raise ValueError("original and counterfactual z must be matching vectors")
    if weights.ndim != 2 or weights.shape[1] != len(z_orig):
        raise ValueError("class weights do not match concept width")
    if not (0 <= source_species < weights.shape[0] and
            0 <= donor_species < weights.shape[0]):
        raise ValueError("species index outside class-head rows")
    if set(original_values) != set(spans):
        raise ValueError("original values must name every output block exactly")

    weight_difference = weights[source_species] - weights[donor_species]
    rows: list[dict] = []
    block_gap_sum = 0.0
    for output_part, (lo, hi) in spans.items():
        before = z_orig[lo:hi]
        after = z_cf[lo:hi]
        delta = after - before
        gap_shift = float(weight_difference[lo:hi] @ delta)
        block_gap_sum += gap_shift
        row = {
            "input_part": input_part,
            "output_part": output_part,
            "coordinates": int(hi - lo),
            "mean_absolute_score_change": float(np.mean(np.abs(delta))),
            "maximum_absolute_score_change": float(np.max(np.abs(delta))),
            "binary_flip_fraction": float(np.mean((before > 0) != (after > 0))),
            "class_gap_shift": gap_shift,
            "target_response_delta": np.nan,
            "unchanged_margin_change": np.nan,
            "original_true_value": int(original_values[output_part]),
        }
        if output_part == input_part:
            if not (0 <= source_value < hi - lo and 0 <= donor_value < hi - lo):
                raise ValueError("source/donor value outside changed block")
            before_margin = before[donor_value] - before[source_value]
            after_margin = after[donor_value] - after[source_value]
            row["target_response_delta"] = float(after_margin - before_margin)
        else:
            true_local = int(original_values[output_part])
            row["unchanged_margin_change"] = (
                best_other_margin(after, true_local) -
                best_other_margin(before, true_local)
            )
        rows.append(row)

    direct_gap_shift = float(weight_difference @ (z_cf - z_orig))
    if not np.isclose(block_gap_sum, direct_gap_shift, atol=1e-9, rtol=0):
        raise RuntimeError(
            f"class-head block decomposition does not close: "
            f"{block_gap_sum} versus {direct_gap_shift}")
    for row in rows:
        row["total_class_gap_shift"] = direct_gap_shift
    return rows


def summarize_pathways(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "input_part", "output_part", "original_image", "render_id",
        "mean_absolute_score_change", "binary_flip_fraction", "class_gap_shift",
        "target_response_delta", "unchanged_margin_change",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"pathway rows lack summary columns: {sorted(missing)}")
    return (rows.groupby(["input_part", "output_part"], sort=False)
            .agg(
                n_rows=("render_id", "size"),
                n_originals=("original_image", "nunique"),
                mean_absolute_score_change=("mean_absolute_score_change", "mean"),
                median_absolute_score_change=("mean_absolute_score_change", "median"),
                mean_binary_flip_fraction=("binary_flip_fraction", "mean"),
                mean_class_gap_shift=("class_gap_shift", "mean"),
                median_class_gap_shift=("class_gap_shift", "median"),
                fraction_class_gap_shift_sourceward=(
                    "class_gap_shift", lambda x: float((x > 0).mean())),
                mean_target_response_delta=("target_response_delta", "mean"),
                mean_unchanged_margin_change=("unchanged_margin_change", "mean"),
            ).reset_index())
