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
import os
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
    pre.add_argument("--curated-koh-root", type=Path)
    pre.add_argument("--curated-neutral-constant-imbalance", action="store_true")
    pre.add_argument("--curated-backbone", choices=("inception_v3", "resnet50"),
                     default="inception_v3")
    pre.add_argument("--curated-require-seed-one", action="store_true")
    own, remaining = pre.parse_known_args()

    curated = Path(__file__).resolve().parents[1]
    koh = own.curated_koh_root or curated / "external" / "ConceptBottleneck"
    if not (koh / "experiments.py").is_file():
        raise SystemExit(f"official Koh entry point missing: {koh / 'experiments.py'}")
    sys.path.insert(0, str(koh))

    if own.curated_require_seed_one:
        try:
            seed_index = remaining.index("--seed")
            koh_seed = int(remaining[seed_index + 1])
        except (ValueError, IndexError):
            raise SystemExit("Koh seed is missing or not an integer")
        if koh_seed != 1:
            raise SystemExit(
                f"seed-one guard rejected Koh seed {koh_seed}; no seed 2/3 is allowed"
            )

    import CUB.config as config  # must happen before importing CUB.train

    config.N_CLASSES = own.curated_num_classes
    if own.curated_neutral_constant_imbalance:
        import CUB.dataset as dataset
        dataset.find_class_imbalance = constant_safe_imbalance
    if own.curated_backbone == "resnet50":
        import CUB.models as koh_models
        from koh_resnet import build_koh_resnet50_joint

        # Fail closed if the pinned Koh constructor no longer has the expected
        # Inception-only boundary.  The replacement is limited to Joint model
        # construction; parser, data, loss, optimizer, scheduler, and loop stay
        # in the pinned Koh repository.
        if "inception_v3" not in koh_models.ModelXtoCtoY.__code__.co_names:
            raise SystemExit("unexpected Koh Joint constructor; refusing ResNet patch")
        koh_models.ModelXtoCtoY = build_koh_resnet50_joint
        # CUB.train copies both N_CLASSES and ModelXtoCtoY with ``from`` imports.
        # Import it only after both values above are set, then prove the copied
        # bindings are exactly the declared class count and ResNet constructor.
        import CUB.train as koh_train
        if koh_train.N_CLASSES != own.curated_num_classes:
            raise SystemExit(
                "Koh train copied the wrong class count: "
                f"{koh_train.N_CLASSES} != {own.curated_num_classes}"
            )
        if koh_train.ModelXtoCtoY is not build_koh_resnet50_joint:
            raise SystemExit("Koh train copied a non-ResNet Joint constructor")
        print(
            "[KOH RESNET IMPORT BOUNDARY PASS] "
            f"classes={koh_train.N_CLASSES} constructor=build_koh_resnet50_joint"
        )

    training_protocol = os.environ.get("KOH_TRAINING_PROTOCOL", "koh_original")
    if training_protocol == "accelerated_v1":
        if own.curated_backbone != "resnet50" or own.curated_num_classes != 50:
            raise SystemExit(
                "accelerated_v1 is approved only for the 50-class FunnyBird "
                "ResNet-50 Joint CBM"
            )
        from koh_accelerated_training import install

        install(koh_train)
    elif training_protocol != "koh_original":
        raise SystemExit(f"unsupported KOH_TRAINING_PROTOCOL={training_protocol!r}")

    forbidden = [entry for entry in sys.path if "minimal_cbm" in entry.replace("\\", "/")]
    if forbidden:
        raise SystemExit(f"minimal_cbm path present in Koh process: {forbidden}")
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_path(str(koh / "experiments.py"), run_name="__main__")


if __name__ == "__main__":
    main()
