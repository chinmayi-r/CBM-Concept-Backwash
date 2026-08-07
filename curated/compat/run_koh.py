#!/usr/bin/env python3
"""Run the pinned Koh ConceptBottleneck trainer with declared dataset sizes.

The upstream CUB trainer hard-codes ``N_CLASSES=200`` in ``CUB.config`` and
binds it into ``CUB.train`` at import time.  FunnyBird (50 classes) and CUB70
(70 classes) therefore require this tiny pre-import adapter.  No upstream
source file is edited; after setting the two dataset constants this delegates
to the pinned repository's own parser and experiment dispatcher.

The two adapter-only flags are removed before the upstream parser sees argv.
All remaining arguments are the official ``experiments.py`` arguments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--curated-num-classes", required=True, type=int)
    pre.add_argument("--curated-num-attributes", required=True, type=int)
    own, remaining = pre.parse_known_args()

    curated = Path(__file__).resolve().parents[1]
    koh = curated / "external" / "ConceptBottleneck"
    if not (koh / "experiments.py").is_file():
        raise SystemExit(f"official Koh entry point missing: {koh / 'experiments.py'}")
    sys.path.insert(0, str(koh))

    import CUB.config as config  # must happen before importing CUB.train

    config.N_CLASSES = own.curated_num_classes
    config.N_ATTRIBUTES = own.curated_num_attributes

    # The official parser reads sys.argv directly.
    sys.argv = [sys.argv[0], *remaining]
    from experiments import parse_arguments, run_experiments
    import numpy as np
    import torch

    dataset, args = parse_arguments()
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)
    run_experiments(dataset, args)


if __name__ == "__main__":
    main()
