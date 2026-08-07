#!/usr/bin/env python3
"""Evaluate one canonical minimal_cbm checkpoint on the untouched final test.

Training uses ``mcbm_selection/test.pkl`` as validation.  This exporter is a
separate, manifest-tracked pass over ``final_test/test.pkl`` so validation
predictions can never be presented as final-test results.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CURATED = Path(__file__).resolve().parents[1]
MCBM = CURATED / "external" / "minimal_cbm"
for path in (MCBM, CURATED / "compat"):
    sys.path.insert(0, str(path))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")


def flat(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array[..., 0] if array.ndim == 3 and array.shape[-1] == 1 else array


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--epoch", required=True, type=int)
    ap.add_argument("--final-test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from src.datasets import get_loader
    from src.helpers import read_config
    from src.models import get_model

    config_group = args.config.split("-")[0]
    cfg = read_config(str(MCBM / "configs" / config_group / args.config))
    data_cfg = dict(cfg["data"])
    data_cfg["pkls_dir"] = str(args.final_test)
    data_cfg["batch_size"] = cfg["training"]["batch_size"]
    loader, model_kwargs, _ = get_loader(
        train=False, seed=args.seed, **data_cfg, return_nuisances=True
    )
    model = get_model(**model_kwargs, **cfg["model"])
    checkpoint = MCBM / "results" / args.config / str(args.seed) / "models" / f"epoch_{args.epoch}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    batches: dict[str, list[torch.Tensor]] = {key: [] for key in ("z", "c_logits", "c_preds", "y_preds", "c", "y")}
    with torch.inference_mode():
        for x, y, c, *_ in loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            output = model(x, c)
            for key in ("z", "c_logits", "c_preds", "y_preds"):
                batches[key].append(output[key].detach().cpu())
            batches["c"].append(c.detach().cpu())
            batches["y"].append(y.detach().cpu())
    arrays = {key: flat(torch.cat(value)) for key, value in batches.items()}

    with (args.final_test / "test.pkl").open("rb") as stream:
        records = pickle.load(stream)
    n_images, n_concepts = arrays["c"].shape
    if len(records) != n_images:
        raise RuntimeError(f"row mismatch: pickle={len(records)} evaluation={n_images}")
    images = [str(row.get("image", row["img_path"])) for row in records]
    frame = pd.DataFrame({
        "image": np.repeat(images, n_concepts),
        "image_index": np.repeat(np.arange(n_images), n_concepts),
        "concept_index": np.tile(np.arange(n_concepts), n_images),
        "y_true": np.repeat(arrays["y"].astype(int), n_concepts),
        "y_pred": np.repeat(arrays["y_preds"].argmax(axis=1).astype(int), n_concepts),
        "c": arrays["c"].reshape(-1),
        "z": arrays["z"].reshape(-1),
        "concept_logit": arrays["c_logits"].reshape(-1),
        "concept_probability": arrays["c_preds"].reshape(-1),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"[FINAL TEST EXPORT SUCCESS] {args.out}: {n_images} images x {n_concepts} concepts")


if __name__ == "__main__":
    main()
