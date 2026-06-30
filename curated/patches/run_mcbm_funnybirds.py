#!/usr/bin/env python3
"""
curated/patches/run_mcbm_funnybirds.py

Drop-in replacement for external/minimal_cbm/bin/train.py that adds FunnyBirds
support. minimal_cbm's dataset dispatcher (src/datasets/__init__.py:get_loader)
is a hardcoded if/elif with no FunnyBirds branch, and per curated/README.md
nothing under external/ may be edited. So this wrapper monkey-patches
src.datasets.get_loader to add a "FUNNYBIRDS" case (delegating to
curated.compat.funnybirds_mcbm_dataset.get_funnybirds) before src.experiments
is ever imported -- `from src.datasets import get_loader` in
src/experiments/train.py binds a name at import time, so the patch must land
first. Everything else (TrainExperiment, InterveneExperiment, config parsing)
runs exactly as in the official bin/train.py.

Usage (run from inside external/minimal_cbm, same convention as bin/train.py):
    python3 ../../patches/run_mcbm_funnybirds.py funnybirds-mcbm -s <seed>

config_file is a BARE name (no path, no .yaml) -- BaseExperiment.__init__
(src/experiments/base.py) hardcodes config lookup to
<minimal_cbm_root>/configs/<config_file.split('-')[0]>/<config_file>.yaml, so
the rendered YAML must physically exist at
external/minimal_cbm/configs/funnybirds/funnybirds-mcbm.yaml (see
curated/train/configs/funnybirds-mcbm.yaml + mcbm_funnybirds.sh, which
sed-renders __CURATED_DATA__/__GAMMA__ placeholders into that location before
invoking this script).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# external/minimal_cbm must be on sys.path for `import src.*` to resolve --
# the caller is expected to `cd` there first (curated/train/mcbm_funnybirds.sh
# does this), but add cwd defensively in case it isn't.
if "" not in sys.path:
    sys.path.insert(0, "")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import src.datasets as _sd  # noqa: E402  (must precede src.experiments import)
from curated.compat.funnybirds_mcbm_dataset import get_funnybirds  # noqa: E402

_original_get_loader = _sd.get_loader


def _patched_get_loader(dataset, **kwargs):
    if dataset.upper() == "FUNNYBIRDS":
        return get_funnybirds(**kwargs)
    return _original_get_loader(dataset, **kwargs)


_sd.get_loader = _patched_get_loader

# Only safe to import after the patch above is in place.
from src.experiments import TrainExperiment, InterveneExperiment  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train MCBM (FunnyBirds-patched)")
    parser.add_argument("config_file", nargs="?", type=str, help="configuration file")
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for initialization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train = TrainExperiment(**vars(args), wandb_key="")
    train.run()

    intervene = InterveneExperiment(**vars(args), wandb_key="")
    intervene.run()
