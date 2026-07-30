#!/usr/bin/env python3
"""Fail-closed audit for the FunnyBirds standard-versus-RLv2 comparison.

The causal comparison is valid only if the two regimes use the same image and
class-label records for training and validation. Their concept labels may differ.

With no extra flags, this checks the base and ``*_trainval`` pickle directories.
With ``--predictions``, it also reads the generated minimal_cbm configs and the
epoch-100 prediction dumps, verifies their saved targets against the configured
evaluation pickle, and reports the actual evaluation population and accuracy.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def config_pairs(rl_tag: str):
    return {
        "CBM": ("funnybirds-cbm", f"funnybirds-cbm-{rl_tag}"),
        "MCBM-g0": ("funnybirds-mcbm-g0", f"funnybirds-mcbm-{rl_tag}-g0"),
        "MCBM-g0p1": ("funnybirds-mcbm-g0p1", f"funnybirds-mcbm-{rl_tag}-g0p1"),
    }


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def image_key(record: dict) -> str:
    value = record.get("image")
    if value is not None:
        return str(value).replace("\\", "/")
    return str(record["img_path"]).replace("\\", "/")


def identity(record: dict):
    return image_key(record), int(record["class_label"])


def compare_records(standard, relabeled, label: str) -> int:
    errors = 0
    if len(standard) != len(relabeled):
        print(f"[FAIL] {label}: lengths differ: {len(standard)} != {len(relabeled)}")
        return 1

    standard_ids = [identity(row) for row in standard]
    relabeled_ids = [identity(row) for row in relabeled]
    if standard_ids != relabeled_ids:
        first = next(
            (i for i, pair in enumerate(zip(standard_ids, relabeled_ids))
             if pair[0] != pair[1]),
            None,
        )
        print(f"[FAIL] {label}: image/class order differs; first mismatch index={first}")
        if first is not None:
            print(f"       standard={standard_ids[first]}")
            print(f"       RLv2={relabeled_ids[first]}")
        errors += 1

    standard_record_ids = [row.get("id") for row in standard]
    relabeled_record_ids = [row.get("id") for row in relabeled]
    if standard_record_ids != relabeled_record_ids:
        print(f"[FAIL] {label}: record IDs differ")
        errors += 1

    allowed_difference = {"attribute_label"}
    for index, (standard_row, relabeled_row) in enumerate(zip(standard, relabeled)):
        keys = set(standard_row) | set(relabeled_row)
        unexpected = [
            key for key in keys
            if key not in allowed_difference
            and standard_row.get(key) != relabeled_row.get(key)
        ]
        if unexpected:
            print(
                f"[FAIL] {label}: record {index} differs outside attribute_label: "
                f"{sorted(unexpected)}"
            )
            errors += 1
            break

    changed = sum(
        a.get("attribute_label") != b.get("attribute_label")
        for a, b in zip(standard, relabeled)
    )
    print(
        f"[{'PASS' if errors == 0 else 'FAIL'}] {label}: "
        f"n={len(standard)}, concept-label changes={changed}"
    )
    return errors


def semantic_differences(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(path + (key,))
            else:
                differences.extend(
                    semantic_differences(left[key], right[key], path + (key,))
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(semantic_differences(a, b, path + (index,)))
        return differences
    return [] if left == right else [path]


def gamma_tag(value: str) -> str:
    number = float(value)
    if number == 0:
        return "0"
    text = str(number).replace(".", "p")
    return text[:-2] if text.endswith("p0") else text


def audit_config_parity(
    repo: Path, curated_data: Path, rl_tag: str, gammas: list[str],
    include_cbm: bool = True,
) -> int:
    import yaml

    config_dir = (
        repo / "external" / "minimal_cbm" / "configs" / "funnybirds"
    )
    pairs = (
        [("funnybirds-cbm", f"funnybirds-cbm-{rl_tag}")]
        if include_cbm else []
    ) + [
            (
                f"funnybirds-mcbm-g{gamma_tag(gamma)}",
                f"funnybirds-mcbm-{rl_tag}-g{gamma_tag(gamma)}",
            )
            for gamma in gammas
    ]
    expected_pkls = str(curated_data / "funnybirds_processed_rl_trainval")
    errors = 0
    for standard_name, matched_name in pairs:
        standard_path = config_dir / f"{standard_name}.yaml"
        matched_path = config_dir / f"{matched_name}.yaml"
        if not standard_path.is_file() or not matched_path.is_file():
            print(f"[FAIL] missing config pair: {standard_path}, {matched_path}")
            errors += 1
            continue
        standard = yaml.safe_load(standard_path.read_text())
        matched = yaml.safe_load(matched_path.read_text())
        differences = semantic_differences(standard, matched)
        expected = [("data", "pkls_dir")]
        path_ok = matched["data"]["pkls_dir"] == expected_pkls
        passed = differences == expected and path_ok
        print(
            f"[{'PASS' if passed else 'FAIL'}] CONFIG {standard_name} vs "
            f"{matched_name}: differences={differences}, "
            f"matched_pkls={matched['data']['pkls_dir']}"
        )
        if not passed:
            errors += 1
    return errors


def audit_pickle_parity(curated_data: Path) -> int:
    standard_base = curated_data / "funnybirds_processed"
    relabeled_base = curated_data / "funnybirds_processed_rl"
    standard_tv = curated_data / "funnybirds_processed_trainval"
    relabeled_tv = curated_data / "funnybirds_processed_rl_trainval"

    required = [
        standard_base / "train.pkl",
        standard_base / "test.pkl",
        relabeled_base / "train.pkl",
        relabeled_base / "test.pkl",
        standard_tv / "train.pkl",
        standard_tv / "test.pkl",
        relabeled_tv / "train.pkl",
        relabeled_tv / "test.pkl",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        for path in missing:
            print(f"[FAIL] missing {path}")
        return len(missing)

    errors = 0
    for split in ("train", "test"):
        errors += compare_records(
            load_pickle(standard_base / f"{split}.pkl"),
            load_pickle(relabeled_base / f"{split}.pkl"),
            f"base/{split}",
        )
        errors += compare_records(
            load_pickle(standard_tv / f"{split}.pkl"),
            load_pickle(relabeled_tv / f"{split}.pkl"),
            f"trainval/{split}",
        )

    standard_train = load_pickle(standard_tv / "train.pkl")
    standard_val = load_pickle(standard_tv / "test.pkl")
    train_keys = {identity(row) for row in standard_train}
    val_keys = {identity(row) for row in standard_val}
    overlap = train_keys & val_keys
    if overlap:
        print(f"[FAIL] standard train/validation overlap: {len(overlap)} records")
        errors += 1
    else:
        print(
            f"[PASS] standard split disjoint: train={len(train_keys)}, "
            f"validation={len(val_keys)}"
        )

    return errors


def find_config(repo: Path, prefix: str) -> Path:
    return repo / "external" / "minimal_cbm" / "configs" / "funnybirds" / f"{prefix}.yaml"


def audit_predictions(repo: Path, epoch: int, seed: int, rl_tag: str) -> int:
    try:
        import torch
        import yaml
    except ImportError as exc:
        print(f"[FAIL] --predictions needs torch and PyYAML: {exc}")
        return 1

    errors = 0
    summaries = {}
    pairs = config_pairs(rl_tag)
    for model, prefixes in pairs.items():
        for labels, prefix in zip(("standard", "RLv2"), prefixes):
            config_path = find_config(repo, prefix)
            prediction_path = (
                repo / "external" / "minimal_cbm" / "results" / prefix /
                str(seed) / "predictions" / f"epoch_{epoch}.pth"
            )
            if not config_path.is_file() or not prediction_path.is_file():
                print(f"[FAIL] missing config or prediction: {config_path}, {prediction_path}")
                errors += 1
                continue

            config = yaml.safe_load(config_path.read_text())
            pkls_dir = Path(config["data"]["pkls_dir"])
            evaluation = load_pickle(pkls_dir / "test.pkl")
            expected_y = torch.tensor(
                [int(row["class_label"]) for row in evaluation], dtype=torch.long
            )
            dump = torch.load(prediction_path, map_location="cpu", weights_only=False)
            saved_y = torch.as_tensor(dump["y"]).reshape(-1).long()
            y_preds = torch.as_tensor(dump["y_preds"])
            predicted_y = y_preds.argmax(-1).reshape(-1).long()

            target_match = (
                len(saved_y) == len(expected_y) and torch.equal(saved_y, expected_y)
            )
            if not target_match:
                print(
                    f"[FAIL] {model}/{labels}: saved y does not match "
                    f"{pkls_dir / 'test.pkl'}"
                )
                errors += 1
            accuracy = float((predicted_y == saved_y).float().mean())
            keys = [identity(row) for row in evaluation]
            summaries[(model, labels)] = {
                "pkls_dir": str(pkls_dir),
                "n": len(evaluation),
                "keys": keys,
                "accuracy": accuracy,
            }
            print(
                f"[INFO] {model:9s} {labels:8s} epoch={epoch} n={len(evaluation):5d} "
                f"accuracy={accuracy:.4f} pkls={pkls_dir}"
            )

    for model in pairs:
        standard = summaries.get((model, "standard"))
        relabeled = summaries.get((model, "RLv2"))
        if standard is None or relabeled is None:
            continue
        same_population = standard["keys"] == relabeled["keys"]
        print(
            f"[{'PASS' if same_population else 'FAIL'}] {model}: "
            f"standard/RLv2 evaluation populations "
            f"{'match' if same_population else 'DIFFER'}"
        )
        if not same_population:
            errors += 1
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-data", required=True, type=Path)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="curated repository directory",
    )
    parser.add_argument("--predictions", action="store_true")
    parser.add_argument("--configs", action="store_true")
    parser.add_argument("--skip-cbm", action="store_true")
    parser.add_argument(
        "--gammas", nargs="*", default=["0", "0.1", "0.3", "1", "3", "5"]
    )
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--rl-tag",
        default="rlv2",
        help="RLv2 result tag to audit; use rlv2 for old checkpoints or "
             "rlv2matched for corrected matched-split checkpoints",
    )
    args = parser.parse_args()

    errors = audit_pickle_parity(args.curated_data)
    if args.configs:
        errors += audit_config_parity(
            args.repo, args.curated_data, args.rl_tag, args.gammas,
            include_cbm=not args.skip_cbm,
        )
    if args.predictions:
        errors += audit_predictions(args.repo, args.epoch, args.seed, args.rl_tag)
    if errors:
        print(f"AUDIT FAILED: {errors} parity error(s)")
        return 1
    print("AUDIT PASSED: image/class populations match; only concept labels may differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
