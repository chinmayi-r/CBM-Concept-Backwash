#!/usr/bin/env python3
"""Run the OFFICIAL minimal_cbm TrainExperiment for the curated pipeline.

  1. FunnyBirds loader: dispatched by an EXPLICIT elif in src/datasets/get_loader,
     applied from curated/patches/minimal_cbm.patch by curated/setup.sh (a small,
     tracked, citable edit -- no runtime monkeypatch). The loader body lives in
     curated/compat/mcbm_funnybirds.py, which this runner puts on sys.path.
  2. wandb offline: minimal_cbm hardcodes an author's wandb key and inits ONLINE
     (TrainExperiment.wandb_offline = False). On a cluster that fails or leaks to
     a stranger's account. We force offline via env + the class attribute.

CUB200 is native to minimal_cbm and needs no patch (the FUNNYBIRDS branch is not
taken).

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


def _check_funnybirds_patch():
    # FunnyBirds dispatch is an explicit elif in src/datasets/get_loader, applied
    # by curated/setup.sh from patches/minimal_cbm.patch (no runtime monkeypatch).
    # Fail loud with the fix instead of a bare ValueError if it wasn't applied.
    import inspect
    from src.datasets import get_loader
    if "FUNNYBIRDS" not in inspect.getsource(get_loader):
        sys.exit("[run_mcbm] minimal_cbm is not patched for FUNNYBIRDS.\n"
                 "           Run:  bash curated/setup.sh   (applies patches/minimal_cbm.patch)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config basename WITHOUT .yaml (e.g. funnybirds-mcbm-g30)")
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("--no-parallel", action="store_true",
                    help="disable ModelParallel (single GPU / debugging)")
    args = ap.parse_args()

    # The upstream experiment stores ``seed`` but does not seed model
    # initialization itself.  Do that in the thin runner so seed 1/2/3 are
    # actual reproducible experimental seeds rather than labels on uncontrolled
    # random starts.  This does not change the model or loss implementation.
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _check_funnybirds_patch()

    from src.experiments import TrainExperiment
    # neutralize the hardcoded-key online init (repo defaults wandb_offline=False)
    TrainExperiment.wandb_offline = True

    train = TrainExperiment(
        config_file=args.config,
        seed=args.seed,
        wandb_key=os.environ.get("WANDB_API_KEY", "offline"),
        parallel=not args.no_parallel,
    )
    if args.config.startswith("cub70-") and train.model_kwargs.get("dim_y") != 70:
        sys.exit("[run_mcbm] CUB70 must build a 70-way task head, but loader returned "
                 f"dim_y={train.model_kwargs.get('dim_y')}. Re-run: bash setup.sh "
                 "to apply the curated minimal_cbm CUB70 patch.")
    # Smoke-test override: MCBM_MAX_EPOCHS caps epochs without editing the config,
    # so a short GPU slot can validate the z snap to +-3. n_epochs/save_epochs are
    # read inside run() (not at construction), so overriding cfg here is safe.
    cap = os.environ.get("MCBM_MAX_EPOCHS")
    if cap:
        cap = int(cap)
        train.cfg["training"]["n_epochs"] = cap
        train.cfg["training"]["save_epochs"] = min(cap, train.cfg["training"]["save_epochs"])
        print(f"[run_mcbm] MCBM_MAX_EPOCHS set -> capping to {cap} epochs (smoke test)")
    train.run()
    print(f"[run_mcbm] done: config={args.config} seed={args.seed}")
    print(f"[run_mcbm] results under: {MCBM/'results'/args.config/str(args.seed)}")


if __name__ == "__main__":
    main()
