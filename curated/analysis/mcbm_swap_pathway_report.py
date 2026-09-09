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


def _cache_identity(swaps: pd.DataFrame, gamma: float, curated_repo: Path) -> str:
    checkpoint = curated_repo / "external/minimal_cbm/results" / (
        f"funnybirds-mcbm-g{checkpoint_tag(gamma)}"
    ) / "1/models/epoch_100.pt"
    config = curated_repo / "external/minimal_cbm/configs/funnybirds" / (
        f"funnybirds-mcbm-g{checkpoint_tag(gamma)}.yaml"
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

    cache = curated_data / "mcbm_notebook03_original_replay" / _cache_identity(
        swaps, gamma, curated_repo
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
        f"funnybirds-mcbm-g{checkpoint_tag(gamma)}"
    ) / "1/models/epoch_100.pt"
    if cached is None:
        model, width = load_model(
            f"funnybirds-mcbm-g{checkpoint_tag(gamma)}", 1, 100, "cuda"
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
) -> pd.DataFrame:
    """Compute the matched encoder movement and learned-head conversion per row."""
    n = len(swaps)
    if any(value.shape != (n, 26) for value in (h_orig, h_cf, z_orig, z_cf)):
        raise ValueError("pathway arrays must all have shape [swap rows, 26]")
    if len(strict_matched_replay) != n:
        raise ValueError("strict replay mask length mismatch")
    rows = []
    for i, row in enumerate(swaps.itertuples()):
        lo, hi = spans[row.part]
        source = lo + int(row.var_src)
        donor = lo + int(row.var_donor)
        h_donor_gain = float(h_cf[i, donor] - h_orig[i, donor])
        h_source_decrease = float(h_orig[i, source] - h_cf[i, source])
        z_donor_gain = float(z_cf[i, donor] - z_orig[i, donor])
        z_source_decrease = float(z_orig[i, source] - z_cf[i, source])
        h_exact = int(np.argmax(h_cf[i, lo:hi]))
        z_exact = int(np.argmax(z_cf[i, lo:hi]))
        h_success = h_exact == int(row.var_donor)
        z_success = z_exact == int(row.var_donor)
        rows.append(dict(
            gamma=float(row.gamma) if hasattr(row, "gamma") else np.nan,
            part=row.part, render_id=row.render_id, original_image=row.orig_render_id,
            source_value=int(row.var_src), donor_value=int(row.var_donor),
            strict_matched_replay=bool(strict_matched_replay[i]),
            h_donor_gain=h_donor_gain,
            h_source_decrease=h_source_decrease,
            h_response=h_donor_gain + h_source_decrease,
            z_donor_gain=z_donor_gain,
            z_source_decrease=z_source_decrease,
            z_response=z_donor_gain + z_source_decrease,
            h_final_margin=float(h_cf[i, donor] - h_cf[i, source]),
            z_final_margin=float(z_cf[i, donor] - z_cf[i, source]),
            h_exact_donor_recognized=h_success,
            z_exact_donor_recognized=z_success,
            q_breaks_h_exact_success=bool(h_success and not z_success),
            q_repairs_h_exact_failure=bool((not h_success) and z_success),
            donor_path_sign_preserved=bool(h_donor_gain * z_donor_gain > 0),
            source_path_sign_preserved=bool(h_source_decrease * z_source_decrease > 0),
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
            records.append(dict(
                gamma=gamma, part=part, population=population, n_rows=len(q),
                n_originals=q.original_image.nunique(),
                mean_h_donor_gain=q.h_donor_gain.mean(),
                mean_h_source_decrease=q.h_source_decrease.mean(),
                mean_h_response=q.h_response.mean(),
                h_response_positive_rate=(q.h_response > 0).mean(),
                h_exact_donor_recognition=q.h_exact_donor_recognized.mean(),
                mean_z_donor_gain=q.z_donor_gain.mean(),
                mean_z_source_decrease=q.z_source_decrease.mean(),
                mean_z_response=q.z_response.mean(),
                z_response_positive_rate=(q.z_response > 0).mean(),
                z_exact_donor_recognition=q.z_exact_donor_recognized.mean(),
                q_breaks_h_success_rate=q.q_breaks_h_exact_success.mean(),
                q_repairs_h_failure_rate=q.q_repairs_h_exact_failure.mean(),
                donor_path_sign_preserved=q.donor_path_sign_preserved.mean(),
                source_path_sign_preserved=q.source_path_sign_preserved.mean(),
            ))
    return pd.DataFrame(records)


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
        ("mean_h_response", "A · Encoder movement in h", "mean h donorward movement", None, None, "coolwarm"),
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
    args = parser.parse_args()
    curated_repo = Path(__file__).resolve().parents[1]
    curated_data = Path(os.environ["CURATED_DATA"])
    output = args.output or curated_data / "mcbm_swap_pathway_v1"
    output.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(curated_repo / "data/funnybirds"))
    import funnybirds_concepts as fbc
    fb_root = Path(os.environ.get("FUNNYBIRDS_ROOT", curated_data / "FunnyBirds"))
    spans = fbc.group_slices(fbc.load_parts(fb_root))

    print("GOAL: locate MCBM swap-response loss at image->h versus h->q(h)=z", flush=True)
    print("WORK: frozen inference on 250 originals per gamma; no training and no Slurm", flush=True)
    all_rows = []
    for gamma in GAMMAS:
        csv_path = curated_data / "swap_fixed_v2_attempt2" / (
            f"funnybirds-mcbm-g{checkpoint_tag(gamma)}-s1.csv"
        )
        swaps = pd.read_csv(csv_path).assign(gamma=gamma)
        if len(swaps) != 5000 or set(swaps.part) != set(ORDER):
            raise ValueError(f"unexpected accepted swap population: {csv_path}")
        h_cf = replay_counterfactual_h(
            swaps, gamma, curated_data, curated_repo, require_cache=True
        )
        h_orig, z_orig_replayed, strict_orig = replay_original_h(
            swaps, gamma, curated_data, curated_repo
        )
        checkpoint = curated_repo / "external/minimal_cbm/results" / (
            f"funnybirds-mcbm-g{checkpoint_tag(gamma)}"
        ) / "1/models/epoch_100.pt"
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
        result = pathway_rows(swaps, h_orig, h_cf, z_orig, z_cf, spans, strict_matched)
        all_rows.append(result)
        print(f"gamma={gamma:g}: pathway rows complete", flush=True)

    rows = pd.concat(all_rows, ignore_index=True)
    summary = summarize(rows)
    rows.to_csv(output / "pathway_rows.csv", index=False)
    summary.to_csv(output / "pathway_summary.csv", index=False)
    plot_summary(summary, output / "pathway_summary.png")
    manifest = {
        "status": "ACCEPTED FOR locating encoder-versus-q swap-response loss",
        "rows": len(rows), "gammas": list(GAMMAS), "parts": list(ORDER),
        "training": False,
        "pathway_rows_sha256": sha256(output / "pathway_rows.csv"),
        "pathway_summary_sha256": sha256(output / "pathway_summary.csv"),
        "figure_sha256": sha256(output / "pathway_summary.png"),
        "causal_boundary": (
            "same-image swap localizes forward response loss; it does not isolate "
            "which training objective caused independently trained checkpoints to differ"
        ),
    }
    (output / "SUCCESS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("[SUCCESS]", output / "SUCCESS.json", flush=True)


if __name__ == "__main__":
    main()
