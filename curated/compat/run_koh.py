#!/usr/bin/env python3
"""Run Koh's unchanged trainer with a non-CUB class count.

Koh's CUB trainer hard-codes ``N_CLASSES=200`` and imports that value when
``CUB.train`` is first loaded. FunnyBird (50 classes) and CUB70 (70 classes)
therefore need this one pre-import change. After it, ``experiments.py`` runs as
``__main__``: Koh's parser, seeding, model construction, and training are used
unchanged.

The adapter-only flag is removed before Koh's parser sees ``sys.argv``.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--curated-num-classes", required=True, type=int)
    own, remaining = pre.parse_known_args()

    curated = Path(__file__).resolve().parents[1]
    koh = curated / "external" / "ConceptBottleneck"
    if not (koh / "experiments.py").is_file():
        raise SystemExit(f"official Koh entry point missing: {koh / 'experiments.py'}")
    sys.path.insert(0, str(koh))

    import CUB.config as config  # must happen before importing CUB.train

    config.N_CLASSES = own.curated_num_classes
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_path(str(koh / "experiments.py"), run_name="__main__")


if __name__ == "__main__":
    main()
