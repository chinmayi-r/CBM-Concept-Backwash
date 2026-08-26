#!/usr/bin/env python3
"""Reconcile the completed FunnyBird accelerated seed-1 artifact.

This is intentionally specific to Slurm job 3357208.  It preserves the
predeclared convergence result and records the subsequent limited acceptance;
it does not reinterpret the original gate as passing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


TRAINING_REPOSITORY_SHA = "de5d8903bcda32a591a4f514381bdd0501784ebc"
ORIGINAL_SLURM_JOB_ID = "3357208"


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"ERROR: missing/empty required artifact: {path}")
    return json.loads(path.read_text())


def require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"ERROR: {name} changed: expected {expected!r}, got {actual!r}"
        )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("ERROR: missing/empty completed outputs:\n" + "\n".join(missing))


def image_accuracy(path: Path) -> float:
    frame = pd.read_parquet(path)
    required = {"image", "y_true", "y_pred", "z"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"ERROR: {path} lacks columns {missing}")
    if not torch.isfinite(torch.as_tensor(frame.z.to_numpy(float))).all():
        raise SystemExit(f"ERROR: non-finite raw logits in {path}")
    images = frame[["image", "y_true", "y_pred"]].drop_duplicates("image")
    if len(images) != 500:
        raise SystemExit(f"ERROR: expected 500 test images in {path}, found {len(images)}")
    return float((images.y_true == images.y_pred).mean())


def load_model(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_model(model) -> None:
    if type(model).__name__ != "End2EndModel":
        raise SystemExit(f"ERROR: expected Koh End2EndModel, got {type(model).__name__}")
    if getattr(model, "curated_framework", None) != "koh_joint":
        raise SystemExit("ERROR: checkpoint lacks Koh Joint framework marker")
    if getattr(model, "curated_backbone", None) != "resnet50":
        raise SystemExit("ERROR: checkpoint is not the declared ResNet-50 model")
    if getattr(model, "use_sigmoid", None) or getattr(model, "use_relu", None):
        raise SystemExit("ERROR: class head does not read untransformed raw concept logits")
    first = getattr(model, "first_model", None)
    if type(first).__name__ != "KohResNet50ConceptEncoder":
        raise SystemExit(f"ERROR: wrong image encoder: {type(first).__name__}")
    if len(getattr(first, "main_heads", ())) != 26:
        raise SystemExit("ERROR: expected 26 scalar concept heads")
    forbidden = []
    for module in model.modules():
        qualified = f"{type(module).__module__}.{type(module).__name__}".lower()
        if "inception" in qualified or "minimal_cbm" in qualified or ".mcbm" in qualified:
            forbidden.append(qualified)
    if forbidden:
        raise SystemExit(f"ERROR: forbidden model modules: {sorted(set(forbidden))}")
    for name, value in model.state_dict().items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise SystemExit(f"ERROR: non-finite checkpoint tensor: {name}")


def require_same_parameters(left, right) -> None:
    a, b = left.state_dict(), right.state_dict()
    if a.keys() != b.keys():
        raise SystemExit("ERROR: epoch-100 and final checkpoint state keys differ")
    changed = [name for name in a if not torch.equal(a[name], b[name])]
    if changed:
        raise SystemExit(
            "ERROR: epoch-100 and final parameters differ: " + ", ".join(changed[:5])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--curated-data", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    data_root = args.curated_data.resolve()
    sys.path.insert(0, str(repo / "curated/compat"))
    sys.path.insert(0, str(repo / "curated/external/ConceptBottleneck"))
    out = data_root / "koh_joint_resnet_accelerated_v1/funnybirds/standard/seed1"
    success = out / "SUCCESS.json"
    if success.exists():
        raise SystemExit(f"ERROR: refusing to overwrite existing acceptance manifest: {success}")

    train = data_root / "koh_joint_inputs/funnybirds/standard/train.pkl"
    val = data_root / "koh_joint_inputs/funnybirds/standard/val.pkl"
    test = data_root / "koh_joint_inputs/funnybirds/standard/test.pkl"
    protocol_path = out / "TRAINING_PROTOCOL.json"
    model_preflight = out / "MODEL_PREFLIGHT.json"
    integrity_before = out / "INPUT_INTEGRITY.json"
    integrity_after = out / "INPUT_INTEGRITY_AFTER.json"
    checkpoint_report = out / "CHECKPOINT.json"
    convergence_path = out / "CONVERGENCE.json"
    decision_path = out / "CONVERGENCE_DECISION.json"
    final_model = out / "final_model_1.pth"
    final_test = out / "final_test.parquet"
    milestone_models = [out / f"milestone_epoch_{epoch}.pth" for epoch in ("025", "050", "075", "100")]
    milestone_tests = [out / f"milestone_epoch_{epoch}_test.parquet" for epoch in ("025", "050", "075", "100")]
    required = [
        train, val, test, protocol_path, model_preflight, integrity_before,
        integrity_after, checkpoint_report, convergence_path, final_model,
        final_test, *milestone_models, *milestone_tests,
    ]
    require_files(required)

    protocol = read_json(protocol_path)
    expected_protocol = {
        "status": "PASS", "model_family": "koh_joint_resnet50",
        "training_protocol": "accelerated_v1", "epochs": 100,
        "batch_size": 128, "amp": True, "num_workers": 8,
        "optimizer": "SGD", "start_lr": 0.001, "max_lr": 0.02,
        "min_lr": 0.00002, "warmup_epochs": 5,
    }
    wrong = {key: (protocol.get(key), value) for key, value in expected_protocol.items()
             if protocol.get(key) != value}
    if wrong:
        raise SystemExit(f"ERROR: training protocol mismatch: {wrong}")

    if integrity_before != integrity_after and integrity_before.read_bytes() != integrity_after.read_bytes():
        raise SystemExit("ERROR: input-integrity records changed during training")

    checkpoint = read_json(checkpoint_report)
    expected_checkpoint = {
        "status": "SUCCESS", "framework": "official_koh_conceptbottleneck",
        "model": "Joint", "backbone": "resnet50", "use_sigmoid": False,
        "training_protocol": "accelerated_v1", "dataset": "funnybirds",
        "labels": "standard", "seed": 1, "num_classes": 50,
        "num_attributes": 26,
    }
    wrong = {key: (checkpoint.get(key), value) for key, value in expected_checkpoint.items()
             if checkpoint.get(key) != value}
    if wrong:
        raise SystemExit(f"ERROR: checkpoint report mismatch: {wrong}")

    convergence = read_json(convergence_path)
    if convergence.get("status") != "INCOMPLETE" or convergence.get("assessment") != "NOT_STABLE_75_TO_100":
        raise SystemExit("ERROR: original convergence result is not the recorded incomplete audit")
    checks = convergence.get("predeclared_checks", {})
    failed = sorted(name for name, item in checks.items() if not item.get("pass"))
    if failed != ["task_accuracy"]:
        raise SystemExit(f"ERROR: expected only task_accuracy to miss the gate, found {failed}")
    for name in (
        "macro_concept_balanced_accuracy", "macro_positive_recall",
        "median_label_separation", "median_raw_logit_spread",
    ):
        if not checks.get(name, {}).get("pass"):
            raise SystemExit(f"ERROR: required concept-health check did not pass: {name}")
    metrics = convergence["metrics_by_epoch"]
    task75 = float(metrics["75"]["task_accuracy"])
    task100 = float(metrics["100"]["task_accuracy"])
    require_close(task75, 0.978, "epoch-75 task accuracy")
    require_close(task100, 0.992, "epoch-100 task accuracy")
    require_close(float(checks["task_accuracy"]["delta"]), 0.014000000000000012,
                  "task-accuracy stability delta")
    if task100 <= task75:
        raise SystemExit("ERROR: the recorded task change is not beneficial")
    require_close(image_accuracy(milestone_tests[2]), task75, "epoch-75 parquet accuracy")
    require_close(image_accuracy(milestone_tests[3]), task100, "epoch-100 parquet accuracy")
    require_close(image_accuracy(final_test), task100, "final parquet accuracy")

    final = load_model(final_model)
    epoch100 = load_model(milestone_models[3])
    validate_model(final)
    validate_model(epoch100)
    require_same_parameters(final, epoch100)

    decision = {
        "status": "ACCEPTED FOR SEED-1 STANDARD-CBM MODEL HEALTH AND DOWNSTREAM EVALUATION",
        "decision_kind": "post_hoc_narrow_exception",
        "original_convergence_status": "INCOMPLETE",
        "original_convergence_assessment": "NOT_STABLE_75_TO_100",
        "original_slurm_job_id": ORIGINAL_SLURM_JOB_ID,
        "training_repository_sha": TRAINING_REPOSITORY_SHA,
        "failed_predeclared_checks": ["task_accuracy"],
        "passed_predeclared_concept_health_checks": [
            "macro_concept_balanced_accuracy", "macro_positive_recall",
            "median_label_separation", "median_raw_logit_spread",
        ],
        "epoch_75_task_accuracy": task75,
        "epoch_100_task_accuracy": task100,
        "task_accuracy_change": task100 - task75,
        "rationale": (
            "Training and evaluation completed. The sole missed stability threshold was "
            "a beneficial seven-net-image task-accuracy increase on 500 test images; all "
            "predeclared concept-health checks passed. The original audit remains unchanged."
        ),
        "limitations": (
            "This post-hoc decision does not make the original stability predicate pass, "
            "does not establish grounding, and does not generalize to future runs."
        ),
    }
    temporary = decision_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    temporary.replace(decision_path)

    outputs = [
        final_model, checkpoint_report, final_test, integrity_after,
        *milestone_models, *milestone_tests, convergence_path, decision_path,
    ]
    command = [
        sys.executable, str(repo / "curated/analysis/canonical_manifest.py"), "write",
        "--repo", str(repo),
        "--stage", "koh_joint_funnybirds_standard_s1",
        "--manifest", str(success),
        "--command", "post-hoc reconciliation of completed Slurm job 3357208",
    ]
    for path in (train, val, test, protocol_path, integrity_before, model_preflight):
        command += ["--input", str(path)]
    for path in outputs:
        command += ["--output", str(path)]
    for value in (
        "framework=koh_joint", "backbone=resnet50",
        "training_protocol=accelerated_v1", "dataset=funnybirds",
        "labels=standard", "seed=1", f"training_repository_sha={TRAINING_REPOSITORY_SHA}",
        f"original_slurm_job_id={ORIGINAL_SLURM_JOB_ID}",
        "convergence_decision=post_hoc_narrow_beneficial_task_change",
    ):
        command += ["--meta", value]
    environment = os.environ.copy()
    environment.pop("SLURM_JOB_ID", None)
    subprocess.run(command, check=True, env=environment)
    subprocess.run([
        sys.executable, str(repo / "curated/analysis/canonical_manifest.py"), "verify",
        "--manifest", str(success),
    ], check=True)
    print(f"[KOH ACCELERATED RECONCILIATION ACCEPTED] {success}")


if __name__ == "__main__":
    main()
