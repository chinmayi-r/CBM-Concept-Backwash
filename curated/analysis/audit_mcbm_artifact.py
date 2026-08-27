#!/usr/bin/env python3
"""Check that an existing checkpoint is the declared minimal_cbm MCBM run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True,
                        choices=("funnybirds", "cub70", "cub"))
    parser.add_argument("--labels", required=True,
                        choices=("standard", "rlv2"))
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-base-lr", type=float)
    parser.add_argument("--training-precision", choices=("amp", "fp32"),
                        default="amp")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.config.is_file() or not args.checkpoint.is_file():
        raise SystemExit("missing config or checkpoint")
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("model", {}).get("model_type") != "mcbm":
        raise SystemExit("wrong model_type: expected mcbm")
    actual_gamma = float(cfg["model"]["gamma"])
    if actual_gamma != args.gamma:
        raise SystemExit(f"gamma mismatch: expected {args.gamma}, got {actual_gamma}")
    expected_data = "FUNNYBIRDS" if args.dataset == "funnybirds" else "CUB200"
    if str(cfg.get("data", {}).get("dataset", "")).upper() != expected_data:
        raise SystemExit(
            f"dataset mismatch: expected {expected_data}, got {cfg.get('data', {}).get('dataset')}"
        )
    actual_base_lr = float(cfg["training"]["optimizer"]["base_lr"])
    if (args.expected_base_lr is not None
            and actual_base_lr != args.expected_base_lr):
        raise SystemExit(
            f"base learning rate mismatch: expected {args.expected_base_lr}, "
            f"got {actual_base_lr}"
        )

    saved = torch.load(args.checkpoint, map_location="cpu")
    state = saved.get("model", saved)
    if not isinstance(state, dict) or not state:
        raise SystemExit("checkpoint has no model state")
    bad = [name for name, value in state.items()
           if torch.is_tensor(value) and torch.is_floating_point(value)
           and not torch.isfinite(value).all()]
    if bad:
        raise SystemExit(f"non-finite checkpoint tensors: {bad[:5]}")

    repo = args.repo.resolve()
    minimal = repo / "curated/external/minimal_cbm"
    manifest = {
        "status": "SUCCESS",
        "framework": "official_minimal_cbm_with_declared_patch",
        "minimal_cbm_sha": subprocess.check_output(
            ["git", "-C", str(minimal), "rev-parse", "HEAD"], text=True
        ).strip(),
        "minimal_cbm_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(minimal), "diff", "--binary", "--", "."]
        )).hexdigest(),
        "dataset": args.dataset,
        "labels": args.labels,
        "gamma": args.gamma,
        "seed": args.seed,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "finite_state_tensors": len(state),
        "pkls_dir": str(cfg.get("data", {}).get("pkls_dir")),
        "encoder": cfg.get("model", {}).get("encoder"),
        "beta": cfg.get("model", {}).get("beta"),
        "optimizer": cfg.get("training", {}).get("optimizer"),
        "training_precision": args.training_precision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[MCBM ARTIFACT PASS] {args.out}")


if __name__ == "__main__":
    main()
