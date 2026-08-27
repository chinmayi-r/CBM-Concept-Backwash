#!/usr/bin/env python3
"""Fail-closed audit for the declared accelerated FunnyBird CBM protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CURATED = Path(__file__).resolve().parents[1]
COMPAT = CURATED / "compat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(COMPAT))
    import koh_accelerated_training as accelerated

    report = accelerated.protocol_manifest()
    expected = {
        "training_protocol": "accelerated_v1",
        "epochs": accelerated.EPOCHS,
        "base_schedule_epochs": 100,
        "target_epochs": accelerated.EPOCHS,
        "continuation": accelerated.EPOCHS > 100,
        "continuation_lr": 0.00002 if accelerated.EPOCHS > 100 else None,
        "batch_size": 128,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0004,
        "start_lr": 0.001,
        "max_lr": 0.02,
        "min_lr": 0.00002,
        "warmup_epochs": 5,
        "scheduler": "linear_warmup_then_cosine",
        "amp": True,
        "num_workers": 8,
        "milestone_epochs": list(range(25, accelerated.EPOCHS + 1, 25)),
        "restart_format": "koh_accelerated_epoch_boundary_v1",
    }
    mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"ERROR: accelerated protocol mismatch: {mismatches}")

    lrs = [accelerated.MAX_LR * accelerated.lr_multiplier(epoch)
           for epoch in range(accelerated.EPOCHS)]
    if abs(lrs[0] - 0.001) > 1e-12:
        raise SystemExit(f"ERROR: initial LR is {lrs[0]}")
    if abs(lrs[4] - 0.02) > 1e-12:
        raise SystemExit(f"ERROR: warmup peak LR is {lrs[4]}")
    if abs(lrs[99] - 0.00002) > 1e-12:
        raise SystemExit(f"ERROR: final LR is {lrs[-1]}")
    if any(right > left for left, right in zip(lrs[4:], lrs[5:])):
        raise SystemExit("ERROR: post-warmup LR is not monotone decreasing")

    report.update({
        "status": "PASS",
        "lr_epoch_1": lrs[0],
        "lr_epoch_5": lrs[4],
        "lr_epoch_100": lrs[99],
        "lr_target_epoch": lrs[-1],
        "trainer_sha256": sha256(Path(accelerated.__file__)),
        "launcher_sha256": sha256(COMPAT / "run_koh.py"),
        "architecture_adapter_sha256": sha256(COMPAT / "koh_resnet.py"),
        "model_family": "koh_joint_resnet50",
        "loss": "koh_joint_task_plus_0.01_concept_normalized",
        "class_head": "linear_raw_concept_logits_to_species",
    })
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(json.dumps(report, sort_keys=True))
    print("[KOH ACCELERATED AUDIT PASS]")


if __name__ == "__main__":
    main()
