#!/usr/bin/env python3
"""Prepare resumable read-only source tables for Notebook 03 or 03rl.

This is deliberately separate from nbconvert: the conditional species probes
can take hours on 5,000 ordinary images, so progress remains visible and a
completed gamma is never recomputed.  It performs no scientific training.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CURATED_REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CURATED_REPO / "data/funnybirds"))

from funnybird_followup_diagnostics import (  # noqa: E402
    add_descriptors,
    conditional_information,
    load_label_conflict,
    ordinary_value_recognition,
    prediction_audit,
    value_holdout_audit,
)
from funnybirds_concepts import concept_names, group_slices, load_parts  # noqa: E402
from mcbm_loss_report import (  # noqa: E402
    checkpoint_tag,
    loss_gradient_audit,
    model_name,
    replacement_use,
    task_head,
)
from minimal_cbm_scores import (  # noqa: E402
    concept_logits_from_saved_latent,
    validate_saved_probabilities,
)

GAMMAS = (0.0, 0.1, 0.3, 1.0, 3.0, 5.0)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def require_table(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"empty completed table: {path}")


def load_ordinary(result: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = result / "predictions/epoch_100.pth"
    checkpoint = result / "models/epoch_100.pt"
    if not prediction.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint/export pair missing under {result}")
    saved = torch.load(prediction, map_location="cpu", weights_only=False)
    h = saved["z"].float().reshape(len(saved["z"]), -1)
    c = saved["c"].float().reshape(len(h), -1)
    y = np.asarray(saved["y"]).reshape(-1).astype(int)
    probability = np.asarray(saved["y_preds"]).reshape(len(h), -1)
    z = concept_logits_from_saved_latent(h, checkpoint, c.shape[1]).float()
    validate_saved_probabilities(z, saved["c_preds"])
    arrays = (h.numpy(), z.numpy(), c.numpy().astype(int), y, probability)
    if h.shape[1] != 26 or probability.shape[1] != 50:
        raise ValueError(f"unexpected MCBM dimensions under {result}")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError(f"non-finite ordinary export under {result}")
    return arrays


def load_swaps(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "part", "var_src", "var_donor", "orig_render_id", "sid_src",
        "z_new", "z_old", "z_new_orig", "z_old_orig", "margin",
        "pixel_count_cf",
    }
    missing = required - set(table)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    if len(table) != 5000 or table.orig_render_id.nunique() != 250:
        raise ValueError(f"unexpected matched-swap population: {path}")
    table = table.copy()
    table["m_orig"] = table.z_new_orig - table.z_old_orig
    table["m_cf"] = table.z_new - table.z_old
    table["response_delta"] = table.m_cf - table.m_orig
    table["donor_gain"] = table.z_new - table.z_new_orig
    table["source_decrease"] = table.z_old_orig - table.z_old
    table["controlled_event"] = (table.response_delta > 0) & (table.m_cf < 0)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-prefix", default="funnybirds-mcbm")
    parser.add_argument("--swap-root-name", default="swap_fixed_v2_attempt2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    curated_data = Path(os.environ["CURATED_DATA"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    parts = load_parts(Path(os.environ.get("FUNNYBIRDS_ROOT", curated_data / "FunnyBirds")))
    names = concept_names(parts)
    spans = group_slices(parts)
    conflict = load_label_conflict(curated_data, names, spans)
    gradient_tables = []

    suffixes = (
        "FULL_WIDTH_INFORMATION", "EQUAL_WIDTH_INFORMATION", "HEAD_USE",
        "PREDICTIVE", "VALUE_HOLDOUT", "LOSS_GRADIENTS",
    )
    print("PREFLIGHT ONLY - READ-ONLY DIAGNOSTICS, NO SCIENTIFIC TRAINING", flush=True)
    print(f"model_prefix={args.model_prefix}", flush=True)
    print(f"swap_root={curated_data / args.swap_root_name}", flush=True)
    print(f"output={output}", flush=True)

    for gamma in GAMMAS:
        tag = f"g{checkpoint_tag(gamma)}"
        paths = {suffix: output / f"{tag}_{suffix}.csv" for suffix in suffixes}
        if all(path.is_file() and path.stat().st_size for path in paths.values()):
            for path in paths.values():
                require_table(path)
            gradient_tables.append(pd.read_csv(paths["LOSS_GRADIENTS"]))
            print(f"[REUSE COMPLETE] gamma={gamma:g}: all six tables", flush=True)
            continue

        run = model_name(gamma, args.model_prefix)
        result = CURATED_REPO / "external/minimal_cbm/results" / run / "1"
        swap_path = curated_data / args.swap_root_name / f"{run}-s1.csv"
        h, z, c, y, probability = load_ordinary(result)
        swaps = load_swaps(swap_path)
        print(
            f"gamma={gamma:g}: ordinary={len(y)} swaps={len(swaps)}; "
            "starting conditional probes",
            flush=True,
        )
        progress = lambda message: print(f"gamma={gamma:g}: {message}", flush=True)
        info, equal = conditional_information(z, c, y, spans, progress=progress)
        head = replacement_use(h, c, y, probability, task_head(result / "models/epoch_100.pt"), spans)
        gradients = loss_gradient_audit(h, c, y, gamma, spans, model_prefix=args.model_prefix)
        recognition = ordinary_value_recognition(z, c, spans)
        descriptors = add_descriptors(swaps, conflict, recognition)
        predictive, _ = prediction_audit(descriptors)
        holdout = value_holdout_audit(descriptors)
        tables = {
            "FULL_WIDTH_INFORMATION": info,
            "EQUAL_WIDTH_INFORMATION": equal,
            "HEAD_USE": head,
            "PREDICTIVE": predictive,
            "VALUE_HOLDOUT": holdout,
            "LOSS_GRADIENTS": gradients,
        }
        for suffix, table in tables.items():
            write_csv(table, paths[suffix])
        gradient_tables.append(gradients)
        print(f"[GAMMA COMPLETE] gamma={gamma:g}: six tables saved", flush=True)

    combined = pd.concat(gradient_tables, ignore_index=True)
    write_csv(combined, output / "loss_gradients.csv")
    for gamma in GAMMAS:
        tag = f"g{checkpoint_tag(gamma)}"
        for suffix in suffixes:
            require_table(output / f"{tag}_{suffix}.csv")
    require_table(output / "loss_gradients.csv")
    print("[TABLE PREPARATION PASS] every gamma and required table is complete", flush=True)


if __name__ == "__main__":
    main()
