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
import pickle
import runpy
import sys
from pathlib import Path


def constant_safe_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """Koh's exact imbalance formula, defined neutrally for all-zero targets."""
    with open(pkl_file, "rb") as stream:
        data = pickle.load(stream)
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for row in data:
        labels = row["attribute_label"]
        if multiple_attr:
            for index in range(n_attr):
                n_ones[index] += labels[index]
        elif attr_idx >= 0:
            n_ones[0] += labels[attr_idx]
        else:
            n_ones[0] += sum(labels)
    result = []
    for positives, count in zip(n_ones, total):
        result.append(1.0 if positives == 0 else count / positives - 1)
    return result if multiple_attr else result * n_attr


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--curated-num-classes", required=True, type=int)
    pre.add_argument("--curated-neutral-constant-imbalance", action="store_true")
    own, remaining = pre.parse_known_args()

    curated = Path(__file__).resolve().parents[1]
    koh = curated / "external" / "ConceptBottleneck"
    if not (koh / "experiments.py").is_file():
        raise SystemExit(f"official Koh entry point missing: {koh / 'experiments.py'}")
    sys.path.insert(0, str(koh))

    import CUB.config as config  # must happen before importing CUB.train

    config.N_CLASSES = own.curated_num_classes
    if own.curated_neutral_constant_imbalance:
        import CUB.dataset as dataset
        dataset.find_class_imbalance = constant_safe_imbalance
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_path(str(koh / "experiments.py"), run_name="__main__")


if __name__ == "__main__":
    main()
