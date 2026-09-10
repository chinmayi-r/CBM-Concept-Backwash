#!/usr/bin/env python3
"""Locate where an MCBM controlled-swap response is lost: encoder h or q(h)=z.

This is frozen, read-only inference.  It does not train a model and it does not
alter the accepted swap CSVs.  Counterfactual ``h`` comes from the already
accepted replay cache; only the 250 unique ordinary originals per gamma are
inferred here so that every row has a matched ``h_orig`` and ``h_cf``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mcbm_loss_report import (
    REPLAY_LOGIT_ATOL,
    array,
    checkpoint_tag,
    model_name,
    replay_counterfactual_h,
    sha256,
    softmax,
    task_head,
)
from minimal_cbm_scores import concept_logits_from_saved_latent


GAMMAS = (0.0, 0.1, 0.3, 1.0, 3.0, 5.0)
ORDER = ("tail", "wing", "beak", "foot", "eye")
COLORS = dict(tail="#7B3294", wing="#0080C6", beak="#E66101", foot="#009E73", eye="#CC79A7")
DISCLOSED_REPLAY_CAP = 0.05


def _cache_identity(
    swaps: pd.DataFrame,
    gamma: float,
    curated_repo: Path,
    model_prefix: str = "funnybirds-mcbm",
) -> str:
    checkpoint = curated_repo / "external/minimal_cbm/results" / (
        model_name(gamma, model_prefix)
    ) / "1/models/epoch_100.pt"
    config = curated_repo / "external/minimal_cbm/configs/funnybirds" / (
        f"{model_name(gamma, model_prefix)}.yaml"
    )
    digest = hashlib.sha256(b"MCBM_ORIGINAL_RGB256_CENTER224_IMAGENET_BATCH1_V1")
    digest.update(sha256(checkpoint).encode())
    digest.update(sha256(config).encode())
    digest.update(swaps[[
        "orig_render_id", "image_orig_path", "image_orig_sha256",
        "part", "var_src", "var_donor", "z_old_orig", "z_new_orig",
    ]].to_csv(index=False).encode())
    for source in sorted((curated_repo / "external/minimal_cbm/src/models").rglob("*.py")):
        digest.update(source.read_bytes())
    digest.update((curated_repo / "analysis/grounding_deletion.py").read_bytes())
    return digest.hexdigest()


def replay_original_h(
    swaps: pd.DataFrame,
    gamma: float,
    curated_data: Path,
    curated_repo: Path,
    *,
    model_prefix: str = "funnybirds-mcbm",
    original_replay_root_name: str = "mcbm_notebook03_original_replay",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-aligned original h/z and a strict-replay flag.

    The 0.02 threshold is an engineering replay check, not an effect-size
    threshold.  Rows between 0.02 and 0.05 are disclosed and retained only in
    the explicitly labelled all-row sensitivity; values above 0.05 stop.
    """
    from PIL import Image
    from torchvision import transforms
    from grounding_deletion import load_model, _MEAN, _STD

    required = {
        "image_orig_path", "image_orig_sha256", "orig_render_id", "part",
        "var_src", "var_donor", "z_old_orig", "z_new_orig",
    }
    missing = required - set(swaps.columns)
    if missing:
        raise ValueError(f"swap CSV lacks original-image pathway fields: {sorted(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Original-image MCBM pathway replay requires CUDA; no training is performed")

    mapping_counts = swaps.groupby("image_orig_sha256").image_orig_path.nunique()
    if not mapping_counts.eq(1).all():
        raise ValueError("one original RGB hash maps to multiple paths")
    unique = swaps.drop_duplicates("image_orig_sha256").reset_index(drop=True)
    for row in unique.itertuples():
        if sha256(row.image_orig_path) != row.image_orig_sha256:
            raise ValueError(f"accepted original RGB bytes changed: {row.image_orig_path}")

    cache = curated_data / original_replay_root_name / _cache_identity(
        swaps, gamma, curated_repo, model_prefix
    )
    cache.mkdir(parents=True, exist_ok=True)
    arrays_path = cache / "original_h_z_p.npz"
    marker_path = cache / "SUCCESS.json"
    cached = None
    if arrays_path.is_file() and marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("sha256") == sha256(arrays_path):
            with np.load(arrays_path, allow_pickle=False) as saved:
                cached = {key: saved[key] for key in ("rgb", "h", "z", "p")}
            if (
                cached["h"].shape != (len(unique), 26)
                or cached["z"].shape != (len(unique), 26)
                or cached["p"].shape != (len(unique), 50)
                or not np.array_equal(cached["rgb"], unique.image_orig_sha256.to_numpy(str))
                or not all(np.isfinite(cached[key]).all() for key in ("h", "z", "p"))
            ):
                raise ValueError(f"invalid original replay cache: {arrays_path}")
            print(f"gamma={gamma:g}: reuse verified original-image replay {cache}", flush=True)

    checkpoint = curated_repo / "external/minimal_cbm/results" / (
        model_name(gamma, model_prefix)
    ) / "1/models/epoch_100.pt"
    if cached is None:
        model, width = load_model(
            model_name(gamma, model_prefix), 1, 100, "cuda"
        )
        if width != 26 or type(model).__name__ != "MinimalConceptBottleneckModel":
            raise ValueError("original replay did not construct the official 26-slot MCBM")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        h_values, z_values, p_values = [], [], []
        with torch.inference_mode():
            for number, row in enumerate(unique.itertuples(), 1):
                with Image.open(row.image_orig_path) as image:
                    x = transform(image.convert("RGB")).unsqueeze(0).to("cuda")
                out = model(x, torch.zeros(1, 26, device="cuda"), sampling=False)
                h_values.append(array(out["z"]).reshape(26))
                z_values.append(array(out["c_logits"]).reshape(26))
                p_values.append(array(out["y_preds"]).reshape(50))
                if number % 50 == 0 or number == len(unique):
                    print(
                        f"gamma={gamma:g}: original frozen inference {number}/{len(unique)} images",
                        flush=True,
                    )
        cached = dict(
            rgb=unique.image_orig_sha256.to_numpy(str),
            h=np.stack(h_values), z=np.stack(z_values), p=np.stack(p_values),
        )
        partial = cache / "original_h_z_p.partial.npz"
        np.savez(partial, **cached)
        partial.replace(arrays_path)
        del model
        torch.cuda.empty_cache()

    lookup = {
        str(key): (cached["h"][i], cached["z"][i], cached["p"][i])
        for i, key in enumerate(cached["rgb"])
    }
    import funnybirds_concepts as fbc
    fb_root = Path(os.environ.get("FUNNYBIRDS_ROOT", curated_data / "FunnyBirds"))
    spans = fbc.group_slices(fbc.load_parts(fb_root))
    row_h, row_z, errors = [], [], []
    for row in swaps.itertuples():
        h, z, _ = lookup[str(row.image_orig_sha256)]
        lo, _ = spans[row.part]
        indices = lo + np.array([int(row.var_src), int(row.var_donor)])
        expected = np.array([float(row.z_old_orig), float(row.z_new_orig)])
        error = float(np.max(np.abs(z[indices] - expected)))
        row_h.append(h); row_z.append(z); errors.append(error)
    row_h = np.stack(row_h); row_z = np.stack(row_z); errors = np.asarray(errors)
    if not np.isfinite(row_h).all() or not np.isfinite(row_z).all():
        raise ValueError("non-finite original h/z replay")
    if float(errors.max()) > DISCLOSED_REPLAY_CAP:
        raise ValueError(
            f"original replay exceeds disclosed cap: max={errors.max():.6g} > {DISCLOSED_REPLAY_CAP}"
        )

    head_logits = task_head(checkpoint)(cached["h"])
    head_error = float(np.max(np.abs(softmax(head_logits) - cached["p"])))
    if head_error > 2e-6:
        raise ValueError(f"saved species head disagrees with same-session original replay: {head_error}")
    disclosed = swaps.loc[errors > REPLAY_LOGIT_ATOL, ["render_id", "orig_render_id", "part"]].copy()
    disclosed["max_original_score_error"] = errors[errors > REPLAY_LOGIT_ATOL]
    disclosed.to_csv(cache / "disclosed_rows.csv", index=False)
    marker = {
        "status": "ACCEPTED FOR matched h-to-z pathway analysis",
        "gamma": gamma,
        "rows": len(swaps),
        "unique_original_images": len(unique),
        "sha256": sha256(arrays_path),
        "strict_tolerance": REPLAY_LOGIT_ATOL,
        "disclosed_cap": DISCLOSED_REPLAY_CAP,
        "strict_exceedances": int((errors > REPLAY_LOGIT_ATOL).sum()),
        "max_original_score_error": float(errors.max()),
        "same_session_species_head_max_error": head_error,
        "training": False,
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print("ORIGINAL REPLAY ACCEPTED:", marker, flush=True)
    return row_h, row_z, errors <= REPLAY_LOGIT_ATOL


def pathway_rows(
    swaps: pd.DataFrame,
    h_orig: np.ndarray,
    h_cf: np.ndarray,
    z_orig: np.ndarray,
    z_cf: np.ndarray,
    spans: dict[str, tuple[int, int]],
    strict_matched_replay: np.ndarray,
    h_absent_mean: np.ndarray,
    h_present_mean: np.ndarray,
) -> pd.DataFrame:
    """Compute matched encoder movement and learned-head conversion per row.

    Raw ``h`` coordinates are not directly comparable: each coordinate has its
    own learned q_j reader and may use a different sign or scale. Exact-value
    recognition at the h stage therefore uses ordinary-image label anchors:

        h_evidence_j = (h_j - mean(h_j | c_j=0)) /
                       (mean(h_j | c_j=1) - mean(h_j | c_j=0))

    An ordinary absent value maps to zero and an ordinary present value maps to
    one. A coordinate with indistinguishable anchors is not assigned an h-stage
    exact winner. Raw h movement is still reported in native slot units.
    """
    n = len(swaps)
    if any(value.shape != (n, 26) for value in (h_orig, h_cf, z_orig, z_cf)):
        raise ValueError("pathway arrays must all have shape [swap rows, 26]")
    if len(strict_matched_replay) != n:
        raise ValueError("strict replay mask length mismatch")
    h_absent_mean = np.asarray(h_absent_mean, dtype=float)
    h_present_mean = np.asarray(h_present_mean, dtype=float)
    if h_absent_mean.shape != (26,) or h_present_mean.shape != (26,):
        raise ValueError("h label anchors must each have shape [26]")
    h_separation = h_present_mean - h_absent_mean
    calibrated_h = np.full_like(h_cf, np.nan, dtype=float)
    calibrated_h_orig = np.full_like(h_orig, np.nan, dtype=float)
    usable = np.abs(h_separation) > 1e-8
    calibrated_h[:, usable] = (
        h_cf[:, usable] - h_absent_mean[usable]
    ) / h_separation[usable]
    calibrated_h_orig[:, usable] = (
        h_orig[:, usable] - h_absent_mean[usable]
    ) / h_separation[usable]
    rows = []
    for i, row in enumerate(swaps.itertuples()):
        lo, hi = spans[row.part]
        source = lo + int(row.var_src)
        donor = lo + int(row.var_donor)
        h_donor_gain = float(h_cf[i, donor] - h_orig[i, donor])
        h_source_decrease = float(h_orig[i, source] - h_cf[i, source])
        z_donor_gain = float(z_cf[i, donor] - z_orig[i, donor])
        z_source_decrease = float(z_orig[i, source] - z_cf[i, source])
        h_block = calibrated_h[i, lo:hi]
        h_orig_block = calibrated_h_orig[i, lo:hi]
        h_valid = bool(np.isfinite(h_block).all())
        h_exact = int(np.argmax(h_block)) if h_valid else -1
        z_exact = int(np.argmax(z_cf[i, lo:hi]))
        h_success = h_valid and h_exact == int(row.var_donor)
        z_success = z_exact == int(row.var_donor)
        z_source_winner = z_exact == int(row.var_src)
        z_third_winner = not z_success and not z_source_winner
        z_response = z_donor_gain + z_source_decrease
        z_backwash = z_response > 0 and float(z_cf[i, donor] - z_cf[i, source]) < 0
        calibrated_h_donor_gain = (
            float(h_block[int(row.var_donor)] - h_orig_block[int(row.var_donor)])
            if h_valid else np.nan
        )
        calibrated_h_source_decrease = (
            float(h_orig_block[int(row.var_src)] - h_block[int(row.var_src)])
            if h_valid else np.nan
        )
        rows.append(dict(
            gamma=float(row.gamma) if hasattr(row, "gamma") else np.nan,
            part=row.part, render_id=row.render_id, original_image=row.orig_render_id,
            source_value=int(row.var_src), donor_value=int(row.var_donor),
            strict_matched_replay=bool(strict_matched_replay[i]),
            h_donor_gain=h_donor_gain,
            h_source_decrease=h_source_decrease,
            h_response=h_donor_gain + h_source_decrease,
            calibrated_h_donor_gain=calibrated_h_donor_gain,
            calibrated_h_source_decrease=calibrated_h_source_decrease,
            calibrated_h_response=calibrated_h_donor_gain + calibrated_h_source_decrease,
            z_donor_gain=z_donor_gain,
            z_source_decrease=z_source_decrease,
            z_response=z_response,
            h_final_margin=float(h_cf[i, donor] - h_cf[i, source]),
            h_original_margin=float(h_orig[i, donor] - h_orig[i, source]),
            z_final_margin=float(z_cf[i, donor] - z_cf[i, source]),
            z_original_margin=float(z_orig[i, donor] - z_orig[i, source]),
            h_calibrated_source_evidence=(float(h_block[int(row.var_src)]) if h_valid else np.nan),
            h_calibrated_donor_evidence=(float(h_block[int(row.var_donor)]) if h_valid else np.nan),
            h_calibrated_final_margin=(
                float(h_block[int(row.var_donor)] - h_block[int(row.var_src)])
                if h_valid else np.nan
            ),
            h_exact_valid=h_valid,
            h_exact_winner_value=h_exact,
            z_exact_winner_value=z_exact,
            h_exact_donor_recognized=h_success,
            h_exact_donor_recognized_valid=(float(h_success) if h_valid else np.nan),
            z_exact_donor_recognized=z_success,
            z_exact_source_recognized=z_source_winner,
            z_exact_third_value_wins=z_third_winner,
            z_no_donorward_movement=bool(z_response <= 0),
            z_responded_but_source_above_donor=bool(z_backwash),
            z_third_winner_source_above_donor=bool(
                z_third_winner and float(z_cf[i, source]) > float(z_cf[i, donor])
            ),
            z_third_winner_donor_above_source=bool(
                z_third_winner and float(z_cf[i, donor]) > float(z_cf[i, source])
            ),
            q_breaks_h_exact_success=bool(h_valid and h_success and not z_success),
            q_repairs_h_exact_failure=bool(h_valid and (not h_success) and z_success),
            donor_path_sign_preserved=bool(
                h_valid and calibrated_h_donor_gain * z_donor_gain > 0
            ),
            source_path_sign_preserved=bool(
                h_valid and calibrated_h_source_decrease * z_source_decrease > 0
            ),
        ))
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (gamma, part), group in rows.groupby(["gamma", "part"], sort=False):
        for population, q in (
            ("all rows", group),
            ("strict matched replay", group[group.strict_matched_replay]),
        ):
            if q.empty:
                continue
            h_valid = q[q.h_exact_valid]
            records.append(dict(
                gamma=gamma, part=part, population=population, n_rows=len(q),
                n_originals=q.original_image.nunique(),
                mean_h_donor_gain=q.h_donor_gain.mean(),
                mean_h_source_decrease=q.h_source_decrease.mean(),
                mean_h_response=q.h_response.mean(),
                h_response_positive_rate=(q.h_response > 0).mean(),
                mean_calibrated_h_donor_gain=(
                    h_valid.calibrated_h_donor_gain.mean() if len(h_valid) else np.nan
                ),
                mean_calibrated_h_source_decrease=(
                    h_valid.calibrated_h_source_decrease.mean() if len(h_valid) else np.nan
                ),
                mean_calibrated_h_response=(
                    h_valid.calibrated_h_response.mean() if len(h_valid) else np.nan
                ),
                calibrated_h_response_positive_rate=(
                    (h_valid.calibrated_h_response > 0).mean() if len(h_valid) else np.nan
                ),
                h_exact_valid_rows=len(h_valid),
                h_exact_donor_recognition=(
                    h_valid.h_exact_donor_recognized.mean() if len(h_valid) else np.nan
                ),
                mean_z_donor_gain=q.z_donor_gain.mean(),
                mean_z_source_decrease=q.z_source_decrease.mean(),
                mean_z_response=q.z_response.mean(),
                z_response_positive_rate=(q.z_response > 0).mean(),
                z_exact_donor_recognition=q.z_exact_donor_recognized.mean(),
                z_exact_source_winner_rate=q.z_exact_source_recognized.mean(),
                z_exact_third_winner_rate=q.z_exact_third_value_wins.mean(),
                z_no_donorward_movement_rate=q.z_no_donorward_movement.mean(),
                z_backwash_rate=q.z_responded_but_source_above_donor.mean(),
                z_third_winner_source_above_donor_rate=q.z_third_winner_source_above_donor.mean(),
                z_third_winner_donor_above_source_rate=q.z_third_winner_donor_above_source.mean(),
                q_breaks_h_success_rate=(
                    h_valid.q_breaks_h_exact_success.mean() if len(h_valid) else np.nan
                ),
                q_repairs_h_failure_rate=(
                    h_valid.q_repairs_h_exact_failure.mean() if len(h_valid) else np.nan
                ),
                donor_path_sign_preserved=q.donor_path_sign_preserved.mean(),
                source_path_sign_preserved=q.source_path_sign_preserved.mean(),
            ))
    return pd.DataFrame(records)


def h_label_anchors(prediction_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return ordinary-image absent/present h means and a visible audit table."""
    saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
    if "z" not in saved or "c" not in saved:
        raise ValueError(f"ordinary prediction export lacks h/c tensors: {prediction_path}")
    h = array(saved["z"]).reshape(-1, 26).astype(float)
    c = array(saved["c"]).reshape(-1, 26).astype(int)
    if h.shape != c.shape or not np.isfinite(h).all() or not np.isin(c, [0, 1]).all():
        raise ValueError(f"invalid ordinary h/c arrays: {prediction_path}")
    absent = np.empty(26, dtype=float)
    present = np.empty(26, dtype=float)
    audit = []
    for index in range(26):
        h0, h1 = h[c[:, index] == 0, index], h[c[:, index] == 1, index]
        if not len(h0) or not len(h1):
            raise ValueError(f"coordinate {index} lacks an ordinary label bucket")
        absent[index], present[index] = h0.mean(), h1.mean()
        audit.append(dict(
            concept_index=index,
            absent_n=len(h0), present_n=len(h1),
            absent_mean=absent[index], present_mean=present[index],
            present_minus_absent=present[index] - absent[index],
            usable_for_calibrated_exact=abs(present[index] - absent[index]) > 1e-8,
        ))
    return absent, present, pd.DataFrame(audit)


def original_restored_offtarget_intervention(
    swaps: pd.DataFrame,
    h_orig: np.ndarray,
    h_cf: np.ndarray,
    spans: dict[str, tuple[int, int]],
    forward,
) -> pd.DataFrame:
    """Restore swap-induced off-target h changes and rerun the frozen task head.

    Old and donor coordinates stay at their counterfactual values. Every other
    coordinate belonging to the swapped part is restored to the exact original
    image's value. This isolates the downstream effect of off-target changes
    caused by that swap more directly than a population absent-mean erasure.
    """
    if h_orig.shape != h_cf.shape or h_cf.shape != (len(swaps), 26):
        raise ValueError("hybrid intervention requires aligned [swap rows, 26] h arrays")
    hybrid = h_cf.copy()
    restored_counts = []
    for i, row in enumerate(swaps.itertuples()):
        lo, hi = spans[row.part]
        protected = {lo + int(row.var_src), lo + int(row.var_donor)}
        restored = [index for index in range(lo, hi) if index not in protected]
        hybrid[i, restored] = h_orig[i, restored]
        restored_counts.append(len(restored))
        if not np.array_equal(hybrid[i, list(protected)], h_cf[i, list(protected)]):
            raise AssertionError("hybrid intervention altered source/donor coordinates")
    before = forward(h_cf)
    after = forward(hybrid)
    probabilities_before = softmax(before)
    probabilities_after = softmax(after)
    row_index = np.arange(len(swaps))
    source = swaps.sid_src.to_numpy(int)
    donor = swaps.sid_donor.to_numpy(int)
    gap_before = before[row_index, source] - before[row_index, donor]
    gap_after = after[row_index, source] - after[row_index, donor]
    return pd.DataFrame(dict(
        render_id=swaps.render_id.to_numpy(),
        original_image=swaps.orig_render_id.to_numpy(),
        part=swaps.part.to_numpy(),
        source_value=swaps.var_src.to_numpy(int),
        donor_value=swaps.var_donor.to_numpy(int),
        restored_offtarget_coordinates=restored_counts,
        source_minus_donor_gap_before=gap_before,
        source_minus_donor_gap_after=gap_after,
        swap_induced_offtarget_source_evidence=gap_before - gap_after,
        pairwise_source_to_donor_flip=(gap_before > 0) & (gap_after < 0),
        top1_changed=before.argmax(1) != after.argmax(1),
        mean_probability_mass_moved=.5 * np.abs(
            probabilities_before - probabilities_after
        ).sum(axis=1),
    ))


def summarize_hybrid(rows: pd.DataFrame) -> pd.DataFrame:
    return (rows.groupby(["gamma", "part"], sort=False)
            .agg(
                n_rows=("render_id", "size"),
                n_originals=("original_image", "nunique"),
                mean_swap_induced_offtarget_source_evidence=(
                    "swap_induced_offtarget_source_evidence", "mean"
                ),
                median_swap_induced_offtarget_source_evidence=(
                    "swap_induced_offtarget_source_evidence", "median"
                ),
                fraction_source_evidence_positive=(
                    "swap_induced_offtarget_source_evidence", lambda value: float((value > 0).mean())
                ),
                pairwise_source_to_donor_flip_rate=("pairwise_source_to_donor_flip", "mean"),
                top1_change_rate=("top1_changed", "mean"),
                mean_probability_mass_moved=("mean_probability_mass_moved", "mean"),
            ).reset_index())


def attach_measured_contributors(
    rows: pd.DataFrame,
    swaps: pd.DataFrame,
    visibility: pd.DataFrame,
    concept_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only pre-existing, pre-outcome measurements to pathway rows."""
    keys = ["render_id", "part"]
    swap_fields = swaps[[
        "render_id", "part", "direction", "sid_src", "sid_donor",
        "pixel_count_cf",
    ]].copy()
    if swap_fields.duplicated(keys).any():
        raise ValueError("swap contributor key is not unique")
    visible_column = (
        "corrected_all_instance_pixels"
        if "corrected_all_instance_pixels" in visibility else "pixel_count_cf"
    )
    visible = visibility[["render_id", "part", visible_column]].rename(
        columns={visible_column: "corrected_visible_pixels"}
    )
    if visible.duplicated(keys).any():
        raise ValueError("visibility contributor key is not unique")
    result = rows.merge(swap_fields, on=keys, how="left", validate="one_to_one")
    result = result.merge(visible, on=keys, how="left", validate="one_to_one")
    if result[["direction", "sid_src", "sid_donor", "corrected_visible_pixels"]].isna().any().any():
        raise ValueError("measured contributors do not cover every pathway row")

    required = {"part", "value", "conflict_rate", "species_support", "positive_images"}
    if required - set(concept_factors):
        raise ValueError(f"concept-factor table lacks {sorted(required-set(concept_factors))}")
    factors = concept_factors[list(required)].drop_duplicates(["part", "value"])
    for side in ("source", "donor"):
        renamed = factors.rename(columns={
            "value": f"{side}_value",
            "conflict_rate": f"{side}_conflict_rate",
            "species_support": f"{side}_species_support",
            "positive_images": f"{side}_positive_training_records",
        })
        result = result.merge(
            renamed, on=["part", f"{side}_value"], how="left", validate="many_to_one"
        )
    factor_columns = [
        f"{side}_{name}"
        for side in ("source", "donor")
        for name in ("conflict_rate", "species_support", "positive_training_records")
    ]
    if result[factor_columns].isna().any().any():
        raise ValueError("concept factors do not cover every source/donor value")
    return result


def summarize_values(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep each donor value visible instead of treating a part as one unit."""
    return (rows.groupby(["gamma", "part", "donor_value"], sort=False)
            .agg(
                n_rows=("render_id", "size"),
                n_originals=("original_image", "nunique"),
                donor_species_support=("donor_species_support", "first"),
                donor_positive_training_records=("donor_positive_training_records", "first"),
                donor_conflict_rate=("donor_conflict_rate", "first"),
                mean_corrected_visible_pixels=("corrected_visible_pixels", "mean"),
                mean_z_original_margin=("z_original_margin", "mean"),
                mean_z_donor_gain=("z_donor_gain", "mean"),
                mean_z_source_decrease=("z_source_decrease", "mean"),
                mean_z_response=("z_response", "mean"),
                median_z_final_margin=("z_final_margin", "median"),
                no_donorward_movement_rate=("z_no_donorward_movement", "mean"),
                responded_but_source_above_donor_rate=(
                    "z_responded_but_source_above_donor", "mean"
                ),
                exact_donor_recognition=("z_exact_donor_recognized", "mean"),
                exact_source_winner_rate=("z_exact_source_recognized", "mean"),
                exact_third_winner_rate=("z_exact_third_value_wins", "mean"),
                h_exact_valid_rate=("h_exact_valid", "mean"),
                h_calibrated_exact_donor_recognition=(
                    "h_exact_donor_recognized_valid", "mean"
                ),
            ).reset_index())


def matched_original_species_health(
    swaps: pd.DataFrame, h_orig: np.ndarray, forward, gamma: float
) -> dict:
    """Evaluate every gamma on the identical 250 ordinary source images."""
    selected = ~swaps.image_orig_sha256.duplicated()
    logits = forward(h_orig[selected.to_numpy()])
    truth = swaps.loc[selected, "sid_src"].to_numpy(int)
    if logits.shape != (len(truth), 50):
        raise ValueError("matched ordinary task logits have the wrong shape")
    return dict(
        gamma=gamma,
        n_images=len(truth),
        species_accuracy=float((logits.argmax(1) == truth).mean()),
        mean_true_species_probability=float(softmax(logits)[np.arange(len(truth)), truth].mean()),
    )


def plot_direct_outcomes(summary: pd.DataFrame, output: Path) -> None:
    primary = summary[summary.population.eq("strict matched replay")]
    metrics = [
        ("z_exact_donor_recognition", "A · Exact donor value wins"),
        ("z_exact_source_winner_rate", "B · Old source value wins exactly"),
        ("z_exact_third_winner_rate", "C · A third value wins"),
        ("z_no_donorward_movement_rate", "D · No donorward pair movement"),
        ("z_backwash_rate", "E · Donorward, but source remains above donor"),
        ("h_exact_donor_recognition", "F · Donor wins in label-calibrated h"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for axis, (metric, title) in zip(axes.flat, metrics):
        table = primary.pivot(index="gamma", columns="part", values=metric).reindex(
            index=GAMMAS, columns=ORDER
        )
        _heat(axis, table, title, "fraction", 0, 1, "viridis")
    fig.suptitle("Direct outcome accounting: pairwise movement and exact winner are separate")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("mean_swap_induced_offtarget_source_evidence", "A · Mean class-gap contribution", None, None, "coolwarm"),
        ("fraction_source_evidence_positive", "B · Fraction favouring source", 0, 1, "viridis"),
        ("pairwise_source_to_donor_flip_rate", "C · Source→donor pair flips", 0, 1, "magma"),
        ("mean_probability_mass_moved", "D · Mean probability mass moved", 0, None, "viridis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for axis, (metric, title, vmin, vmax, cmap) in zip(axes, metrics):
        table = summary.pivot(index="gamma", columns="part", values=metric).reindex(
            index=GAMMAS, columns=ORDER
        )
        _heat(axis, table, title, metric, vmin, vmax, cmap)
    fig.suptitle("Frozen species head after restoring only swap-induced off-target h changes")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, table, title, label, vmin=None, vmax=None, cmap="viridis"):
    values = table.to_numpy(float)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
    ax.set_xlabel("part"); ax.set_ylabel("gamma"); ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row,col]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(values[row,col]) and values[row,col] < np.nanmedian(values) else "black")
    plt.colorbar(image, ax=ax, label=label, fraction=.046, pad=.04)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    primary = summary[summary.population.eq("strict matched replay")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [
        ("mean_calibrated_h_response", "A · Label-calibrated movement in h", "mean calibrated donorward movement", None, None, "coolwarm"),
        ("mean_z_response", "B · Movement after q(h)=z", "mean z donorward movement", None, None, "coolwarm"),
        ("h_exact_donor_recognition", "C · Inserted value largest in h block", "fraction", 0, 1, "viridis"),
        ("z_exact_donor_recognition", "D · Inserted value largest after q", "fraction", 0, 1, "viridis"),
        ("q_breaks_h_success_rate", "E · q breaks an h-stage success", "fraction of all swaps", 0, 1, "magma"),
        ("q_repairs_h_failure_rate", "F · q repairs an h-stage failure", "fraction of all swaps", 0, 1, "magma_r"),
    ]
    for ax, (column, title, label, vmin, vmax, cmap) in zip(axes.flat, panels):
        table = primary.pivot(index="gamma", columns="part", values=column).reindex(
            index=GAMMAS, columns=ORDER
        )
        _heat(ax, table, title, label, vmin, vmax, cmap)
    fig.suptitle("MCBM controlled swaps: where does the inserted-part response disappear?", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-prefix", default="funnybirds-mcbm")
    parser.add_argument("--swap-root-name", default="swap_fixed_v2_attempt2")
    parser.add_argument("--counterfactual-replay-root-name", default="mcbm_notebook03_replay")
    parser.add_argument("--original-replay-root-name", default="mcbm_notebook03_original_replay")
    parser.add_argument("--factor-table", type=Path)
    parser.add_argument("--analysis-version", default="mcbm_swap_pathway_v3")
    args = parser.parse_args()
    curated_repo = Path(__file__).resolve().parents[1]
    curated_data = Path(os.environ["CURATED_DATA"])
    output = args.output or curated_data / "mcbm_swap_pathway_v3"
    output.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(curated_repo / "data/funnybirds"))
    import funnybirds_concepts as fbc
    fb_root = Path(os.environ.get("FUNNYBIRDS_ROOT", curated_data / "FunnyBirds"))
    spans = fbc.group_slices(fbc.load_parts(fb_root))

    print("GOAL: locate MCBM swap-response loss at image->h versus h->q(h)=z", flush=True)
    print("WORK: frozen inference on 250 originals per gamma; no training and no Slurm", flush=True)
    all_rows = []
    all_hybrid = []
    all_anchor_audits = []
    matched_health = []
    visibility_path = curated_data / "funnybird_visibility_correction_v1/visibility.csv"
    factor_path = args.factor_table or (
        curated_repo / "review/funnybird_followup_v3_4c7265c/followup3_conflict_response.csv"
    )
    if not visibility_path.is_file() or not factor_path.is_file():
        raise FileNotFoundError(f"required contributor tables: {visibility_path}, {factor_path}")
    visibility = pd.read_csv(visibility_path)
    concept_factors = pd.read_csv(factor_path)
    for gamma in GAMMAS:
        run_name = model_name(gamma, args.model_prefix)
        csv_path = curated_data / args.swap_root_name / f"{run_name}-s1.csv"
        swaps = pd.read_csv(csv_path).assign(gamma=gamma)
        if len(swaps) != 5000 or set(swaps.part) != set(ORDER):
            raise ValueError(f"unexpected accepted swap population: {csv_path}")
        # The canonical report helper validates and reuses an accepted cache
        # when present; unlike the later recovery branch, it has no
        # ``require_cache`` keyword.
        h_cf = replay_counterfactual_h(
            swaps, gamma, curated_data, curated_repo,
            model_prefix=args.model_prefix,
            replay_root_name=args.counterfactual_replay_root_name,
        )
        h_orig, z_orig_replayed, strict_orig = replay_original_h(
            swaps, gamma, curated_data, curated_repo,
            model_prefix=args.model_prefix,
            original_replay_root_name=args.original_replay_root_name,
        )
        checkpoint = curated_repo / "external/minimal_cbm/results" / (
            run_name
        ) / "1/models/epoch_100.pt"
        prediction = checkpoint.parents[1] / "predictions/epoch_100.pth"
        h_absent, h_present, anchor_audit = h_label_anchors(prediction)
        anchor_audit.insert(0, "gamma", gamma)
        all_anchor_audits.append(anchor_audit)
        z_orig = concept_logits_from_saved_latent(
            torch.as_tensor(h_orig), checkpoint, 26
        ).numpy()
        z_cf = concept_logits_from_saved_latent(
            torch.as_tensor(h_cf), checkpoint, 26
        ).numpy()
        head_recovery_error = float(np.max(np.abs(z_orig - z_orig_replayed)))
        if head_recovery_error > 2e-5:
            raise ValueError(f"offline q(h) recovery mismatch at gamma={gamma:g}: {head_recovery_error}")
        cf_errors = []
        for i, row in enumerate(swaps.itertuples()):
            lo, _ = spans[row.part]
            indices = lo + np.array([int(row.var_src), int(row.var_donor)])
            expected = np.array([float(row.z_old), float(row.z_new)])
            cf_errors.append(float(np.max(np.abs(z_cf[i, indices] - expected))))
        cf_errors = np.asarray(cf_errors)
        if float(cf_errors.max()) > DISCLOSED_REPLAY_CAP:
            raise ValueError(
                f"counterfactual replay exceeds disclosed cap at gamma={gamma:g}: "
                f"max={cf_errors.max():.6g} > {DISCLOSED_REPLAY_CAP}"
            )
        strict_matched = strict_orig & (cf_errors <= REPLAY_LOGIT_ATOL)
        print(
            f"gamma={gamma:g}: strict matched rows={strict_matched.sum()}/{len(swaps)}; "
            f"counterfactual strict exceedances={(cf_errors > REPLAY_LOGIT_ATOL).sum()}",
            flush=True,
        )
        result = pathway_rows(
            swaps, h_orig, h_cf, z_orig, z_cf, spans, strict_matched,
            h_absent, h_present,
        )
        result = attach_measured_contributors(
            result, swaps, visibility, concept_factors
        )
        all_rows.append(result)
        hybrid = original_restored_offtarget_intervention(
            swaps, h_orig, h_cf, spans, task_head(checkpoint)
        )
        hybrid.insert(0, "gamma", gamma)
        all_hybrid.append(hybrid)
        matched_health.append(matched_original_species_health(
            swaps, h_orig, task_head(checkpoint), gamma
        ))
        print(f"gamma={gamma:g}: pathway rows complete", flush=True)

    rows = pd.concat(all_rows, ignore_index=True)
    summary = summarize(rows)
    hybrid_rows = pd.concat(all_hybrid, ignore_index=True)
    hybrid_summary = summarize_hybrid(hybrid_rows)
    anchor_audit = pd.concat(all_anchor_audits, ignore_index=True)
    value_summary = summarize_values(rows)
    matched_health = pd.DataFrame(matched_health)
    rows.to_csv(output / "pathway_rows.csv", index=False)
    summary.to_csv(output / "pathway_summary.csv", index=False)
    hybrid_rows.to_csv(output / "original_restored_offtarget_rows.csv", index=False)
    hybrid_summary.to_csv(output / "original_restored_offtarget_summary.csv", index=False)
    anchor_audit.to_csv(output / "h_label_anchor_audit.csv", index=False)
    value_summary.to_csv(output / "per_value_pathway_summary.csv", index=False)
    matched_health.to_csv(output / "matched_original_species_health.csv", index=False)
    plot_summary(summary, output / "pathway_summary.png")
    plot_direct_outcomes(summary, output / "direct_outcome_accounting.png")
    plot_hybrid(hybrid_summary, output / "original_restored_offtarget_summary.png")
    manifest = {
        "status": "ACCEPTED FOR calibrated h-versus-q pathway and frozen-head off-target intervention",
        "analysis_version": args.analysis_version,
        "model_prefix": args.model_prefix,
        "swap_root_name": args.swap_root_name,
        "rows": len(rows), "gammas": list(GAMMAS), "parts": list(ORDER),
        "training": False,
        "pathway_rows_sha256": sha256(output / "pathway_rows.csv"),
        "pathway_summary_sha256": sha256(output / "pathway_summary.csv"),
        "original_restored_offtarget_rows_sha256": sha256(
            output / "original_restored_offtarget_rows.csv"
        ),
        "original_restored_offtarget_summary_sha256": sha256(
            output / "original_restored_offtarget_summary.csv"
        ),
        "h_label_anchor_audit_sha256": sha256(output / "h_label_anchor_audit.csv"),
        "per_value_pathway_summary_sha256": sha256(output / "per_value_pathway_summary.csv"),
        "matched_original_species_health_sha256": sha256(
            output / "matched_original_species_health.csv"
        ),
        "figure_sha256": sha256(output / "pathway_summary.png"),
        "direct_outcome_figure_sha256": sha256(output / "direct_outcome_accounting.png"),
        "hybrid_figure_sha256": sha256(output / "original_restored_offtarget_summary.png"),
        "causal_boundary": (
            "same-image swap localizes forward response loss; it does not isolate "
            "which training objective caused independently trained checkpoints to differ"
        ),
    }
    (output / "SUCCESS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("ORIGINAL-RESTORED OFF-TARGET INTERVENTION:", flush=True)
    print(hybrid_summary.to_string(index=False), flush=True)
    print("MATCHED ORDINARY SOURCE-IMAGE SPECIES HEALTH:", flush=True)
    print(matched_health.to_string(index=False), flush=True)
    print("[SUCCESS]", output / "SUCCESS.json", flush=True)


if __name__ == "__main__":
    main()
