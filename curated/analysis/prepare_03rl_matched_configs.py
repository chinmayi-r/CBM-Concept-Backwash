#!/usr/bin/env python3
"""Create RLv2-matched configs from the exact generated standard configs.

This intentionally does not regenerate settings from templates. Each matched
config is a deep copy of its standard comparator with exactly one semantic
change: ``data.pkls_dir`` points to the relabeled, identity-matched train/val
pickles.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def gamma_tag(value: str) -> str:
    number = float(value)
    if number == 0:
        return "0"
    text = str(number).replace(".", "p")
    return text[:-2] if text.endswith("p0") else text


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


def make_pair(config_dir: Path, standard_name: str, matched_name: str, pkls_dir: Path):
    source = config_dir / f"{standard_name}.yaml"
    target = config_dir / f"{matched_name}.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"missing standard comparator config: {source}")

    standard = yaml.safe_load(source.read_text())
    matched = copy.deepcopy(standard)
    matched["data"]["pkls_dir"] = str(pkls_dir)

    differences = semantic_differences(standard, matched)
    expected = [("data", "pkls_dir")]
    if differences != expected:
        raise RuntimeError(
            f"{standard_name} -> {matched_name}: unexpected config differences "
            f"{differences}; expected {expected}"
        )

    target.write_text(yaml.safe_dump(matched, sort_keys=False))
    reread = yaml.safe_load(target.read_text())
    if semantic_differences(standard, reread) != expected:
        raise RuntimeError(f"written config failed parity check: {target}")
    print(
        f"[CONFIG PARITY PASS] {standard_name} -> {matched_name}; "
        "only data.pkls_dir changed"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-data", required=True, type=Path)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--rl-tag", default="rlv2matched")
    parser.add_argument("--cbm", action="store_true")
    parser.add_argument("--gammas", nargs="*", default=[])
    args = parser.parse_args()

    if not args.cbm and not args.gammas:
        parser.error("request --cbm and/or at least one --gammas value")

    pkls_dir = args.curated_data / "funnybirds_processed_rl_trainval"
    for split in ("train.pkl", "test.pkl"):
        if not (pkls_dir / split).is_file():
            raise FileNotFoundError(pkls_dir / split)

    config_dir = (
        args.repo / "external" / "minimal_cbm" / "configs" / "funnybirds"
    )
    if args.cbm:
        make_pair(
            config_dir,
            "funnybirds-cbm",
            f"funnybirds-cbm-{args.rl_tag}",
            pkls_dir,
        )
    for gamma in args.gammas:
        tag = gamma_tag(gamma)
        make_pair(
            config_dir,
            f"funnybirds-mcbm-g{tag}",
            f"funnybirds-mcbm-{args.rl_tag}-g{tag}",
            pkls_dir,
        )


if __name__ == "__main__":
    main()
