#!/usr/bin/env python3
"""Convert saved minimal_cbm predictions into the normalized CUB70 eval table.

No image rerun is needed. minimal_cbm saves latent `z`, `c_preds`, `c`,
`y_preds`, and `y` at every saved epoch. Because these CBMs use learned concept
heads, this exporter replays those heads on latent `z` to recover exact raw
`c_logits` for the normalized table. Rows remain aligned with test.pkl.
"""
from __future__ import annotations
import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
MCBM = CURATED / "external" / "minimal_cbm"
for path in (MCBM, CURATED / "data" / "cub70"):
    sys.path.insert(0, str(path))

from cub70_parts import attribute_to_part
from relabel_cub_with_cub70 import CUB_USED_ATTRIBUTE_IDS
from minimal_cbm_scores import (
    concept_logits_from_saved_latent,
    validate_saved_probabilities,
)


def latest(path: Path) -> Path:
    files = sorted(path.glob("epoch_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"no prediction files in {path}")
    return files[-1]


def selected_concepts(attr_dir: Path, n_groups: int, seed: int):
    names_by_id = {}
    for line in (attr_dir / "attributes.txt").read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            names_by_id[int(fields[0])] = fields[1]
    names = [names_by_id[i] for i in CUB_USED_ATTRIBUTE_IDS]
    ids = list(CUB_USED_ATTRIBUTE_IDS)
    groups = {}
    for j, name in enumerate(names):
        groups.setdefault(name.split("::")[0], []).append(j)
    group_list = list(groups.values())
    if n_groups > len(group_list):
        raise ValueError(f"asked for {n_groups} groups but CUB has {len(group_list)}")
    random.seed(seed)
    selected_groups = sorted(random.sample(range(len(group_list)), n_groups))
    idx = [j for gi, group in enumerate(group_list) if gi in selected_groups for j in group]
    return [names[j] for j in idx], [ids[j] for j in idx]


def flat(x):
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return a.reshape(a.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.helpers import read_config
    prefix = args.config.split("-")[0]
    cfg = read_config(str(MCBM / "configs" / prefix / args.config))
    pred_dir = MCBM / "results" / args.config / str(args.seed) / "predictions"
    pred_path = pred_dir / f"epoch_{args.epoch}.pth" if args.epoch else latest(pred_dir)
    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    records = pickle.loads((Path(cfg["data"]["pkls_dir"]) / "test.pkl").read_bytes())
    names, attribute_ids = selected_concepts(
        Path(cfg["data"]["attr_dir"]),
        int(cfg["data"]["n_groups_concepts"]), args.seed
    )

    prob, gt = flat(pred["c_preds"]), flat(pred["c"])
    model_path = pred_dir.parent / "models" / f"{pred_path.stem}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"prediction exists but matching checkpoint is missing: {model_path}"
        )
    logits_t = concept_logits_from_saved_latent(
        pred["z"], model_path, n_concepts=len(names)
    )
    head_error = validate_saved_probabilities(logits_t, pred["c_preds"])
    z = flat(logits_t)
    y_true = flat(pred["y"]).reshape(-1).astype(int)
    y_logits = flat(pred["y_preds"])
    y_pred = y_logits.argmax(1)
    n = len(records)
    if not (z.shape == prob.shape == gt.shape == (n, len(names))):
        raise ValueError(f"shape mismatch: records={n}, names={len(names)}, "
                         f"z={z.shape}, prob={prob.shape}, gt={gt.shape}")

    rows = []
    for i, record in enumerate(records):
        image = Path(record["img_path"]).stem
        for j, name in enumerate(names):
            rows.append({
                "image": image, "class_label": int(y_true[i]),
                "concept_idx": j, "concept_name": name,
                "attribute_id": int(attribute_ids[j]),
                "part": attribute_to_part(name) or "",
                "z": float(z[i, j]), "prob": float(prob[i, j]),
                "gt_label": int(gt[i, j]), "pred_label": int(prob[i, j] >= 0.5),
                "y_true": int(y_true[i]), "y_pred": int(y_pred[i]),
            })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(
        f"[CONCEPT-HEAD REPLAY PASS] max |sigmoid(raw_logit)-saved_prob|={head_error:.3g}"
    )
    print(f"wrote {out}: {n} images x {len(names)} concepts from {pred_path.name}")


if __name__ == "__main__":
    main()
