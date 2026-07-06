#!/usr/bin/env python3
"""Run the OFFICIAL minimal_cbm TrainExperiment, with two curated shims applied
at import time (no files inside external/ are modified):

  1. FunnyBirds loader: monkeypatch src.datasets.get_loader so dataset=="FUNNYBIRDS"
     dispatches to curated/compat/mcbm_funnybirds.get_funnybirds. Must happen
     BEFORE `from src.experiments import ...` (train.py binds get_loader by name
     at its own import time).
  2. wandb offline: minimal_cbm hardcodes an author's wandb key and inits ONLINE
     (TrainExperiment.wandb_offline = False). On a cluster that fails or leaks to
     a stranger's account. We force offline via env + the class attribute.

CUB200 needs no shim -- it is native to minimal_cbm. Pass its config through and
this runner still works (the FUNNYBIRDS branch is simply not taken).

Usage (config is a BASENAME without .yaml, living in
external/minimal_cbm/configs/<prefix>/<basename>.yaml):
    python run_mcbm.py funnybirds-mcbm-g30 -s 1
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # curated/train
CURATED = HERE.parent                                  # curated
MCBM = CURATED / "external" / "minimal_cbm"
COMPAT = CURATED / "compat"

# minimal_cbm must be importable as `src.*`; compat for `mcbm_funnybirds`
sys.path.insert(0, str(MCBM))
sys.path.insert(0, str(COMPAT))

# force wandb offline BEFORE anything imports wandb
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DISABLED", "true")


def _patch_loader():
    import src.datasets as dsets
    _orig = dsets.get_loader

    def patched(dataset, **kwargs):
        if str(dataset).upper() == "FUNNYBIRDS":
            from mcbm_funnybirds import get_funnybirds
            return get_funnybirds(**kwargs)
        return _orig(dataset, **kwargs)

    dsets.get_loader = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config basename WITHOUT .yaml (e.g. funnybirds-mcbm-g30)")
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("--no-parallel", action="store_true",
                    help="disable ModelParallel (single GPU / debugging)")
    args = ap.parse_args()

    _patch_loader()  # must precede the experiments import

    from src.experiments import TrainExperiment
    # neutralize the hardcoded-key online init
    TrainExperiment.wandb_offline = True

    train = TrainExperiment(
        config_file=args.config,
        seed=args.seed,
        wandb_key=os.environ.get("WANDB_API_KEY", "offline"),
        parallel=not args.no_parallel,
    )
    train.run()
    print(f"[run_mcbm] done: config={args.config} seed={args.seed}")
    print(f"[run_mcbm] results under: {MCBM/'results'/args.config/str(args.seed)}")


if __name__ == "__main__":
    main()
