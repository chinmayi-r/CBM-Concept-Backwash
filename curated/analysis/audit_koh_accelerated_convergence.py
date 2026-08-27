#!/usr/bin/env python3
"""Assess predeclared 25-epoch ordinary-health stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(path: Path) -> dict[str, float]:
    frame = pd.read_parquet(path)
    required = {"image", "y_true", "y_pred", "concept_name", "z", "gt_label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"ERROR: {path} lacks columns {missing}")
    if not np.isfinite(frame["z"].to_numpy(float)).all():
        raise SystemExit(f"ERROR: non-finite raw logits in {path}")

    images = frame.drop_duplicates("image")
    task_accuracy = float((images.y_true == images.y_pred).mean())
    balanced, recalls, spreads, separations = [], [], [], []
    for _, group in frame.groupby("concept_name", sort=False):
        positive = group.gt_label.to_numpy(int) == 1
        negative = ~positive
        prediction = group.z.to_numpy(float) > 0
        if positive.any():
            recall = float(prediction[positive].mean())
            recalls.append(recall)
        else:
            recall = np.nan
        if positive.any() and negative.any():
            specificity = float((~prediction[negative]).mean())
            balanced.append((recall + specificity) / 2)
            separations.append(
                float(np.median(group.z.to_numpy(float)[positive])
                      - np.median(group.z.to_numpy(float)[negative]))
            )
        spreads.append(
            float(np.quantile(group.z, 0.95) - np.quantile(group.z, 0.05))
        )
    return {
        "task_accuracy": task_accuracy,
        "macro_concept_balanced_accuracy": float(np.mean(balanced)),
        "macro_positive_recall": float(np.mean(recalls)),
        "median_raw_logit_spread": float(np.median(spreads)),
        "median_label_separation": float(np.median(separations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-25", required=True, type=Path)
    parser.add_argument("--epoch-50", required=True, type=Path)
    parser.add_argument("--epoch-75", required=True, type=Path)
    parser.add_argument("--epoch-100", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--previous-epoch", type=int)
    parser.add_argument("--current-epoch", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-stable", action="store_true")
    args = parser.parse_args()

    values = {
        "25": metrics(args.epoch_25),
        "50": metrics(args.epoch_50),
        "75": metrics(args.epoch_75),
        "100": metrics(args.epoch_100),
    }
    previous_key, current_key = "75", "100"
    if any(value is not None for value in (
        args.previous, args.current, args.previous_epoch, args.current_epoch
    )):
        if None in (
            args.previous, args.current, args.previous_epoch, args.current_epoch
        ):
            raise SystemExit(
                "ERROR: continuation audit requires previous/current files and epochs"
            )
        if args.current_epoch - args.previous_epoch != 25:
            raise SystemExit("ERROR: convergence checkpoints must be 25 epochs apart")
        previous_key, current_key = str(args.previous_epoch), str(args.current_epoch)
        values[previous_key] = metrics(args.previous)
        values[current_key] = metrics(args.current)
    absolute_limits = {
        "task_accuracy": 0.01,
        "macro_concept_balanced_accuracy": 0.01,
        "macro_positive_recall": 0.015,
    }
    relative_limits = {
        "median_raw_logit_spread": 0.10,
        "median_label_separation": 0.10,
    }
    checks = {}
    for key, limit in absolute_limits.items():
        delta = abs(values[current_key][key] - values[previous_key][key])
        checks[key] = {"kind": "absolute", "delta": delta, "limit": limit,
                       "pass": delta <= limit}
    for key, limit in relative_limits.items():
        baseline = max(abs(values[previous_key][key]), 1e-12)
        delta = abs(values[current_key][key] - values[previous_key][key]) / baseline
        checks[key] = {"kind": "relative", "delta": delta, "limit": limit,
                       "pass": delta <= limit}

    stable = all(item["pass"] for item in checks.values())
    report = {
        "status": "PASS" if stable else "INCOMPLETE",
        "assessment": (
            f"STABLE_{previous_key}_TO_{current_key}"
            if stable else f"NOT_STABLE_{previous_key}_TO_{current_key}"
        ),
        "comparison_epochs": [int(previous_key), int(current_key)],
        "metrics_by_epoch": values,
        "predeclared_checks": checks,
        "note": (
            "This is an ordinary task/concept-health convergence check. "
            "Grounding and backwash still require the fixed-render evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if args.require_stable and not stable:
        raise SystemExit(3)
    print("[KOH ACCELERATED CONVERGENCE PASS]")


if __name__ == "__main__":
    main()
