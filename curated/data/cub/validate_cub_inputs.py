#!/usr/bin/env python3
"""Fast preflight for CUB/CUB70 training inputs and generated config.

Runs before a GPU model is constructed. It catches schema mistakes such as
requesting 28 concept groups when the canonical 112 attributes span only 27.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

from prepare_cub_data import CUB_USED_ATTRIBUTE_IDS, CUB_USED_ATTRIBUTE_INDICES


def load_numbered_names(path: Path) -> dict[int, str]:
    names = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit():
            names[int(fields[0])] = fields[1]
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cub", "cub70"], required=True)
    parser.add_argument("--pkls", type=Path, required=True)
    parser.add_argument("--attr-dir", type=Path, required=True)
    parser.add_argument("--imgs-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.pkls / "train.pkl", args.pkls / "test.pkl",
        args.pkls / "selection_indices.json",
        args.attr_dir / "attributes.txt", args.imgs_dir, args.config,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("CUB preflight missing:\n  " + "\n  ".join(missing))
    selected_schema = json.loads((args.pkls / "selection_indices.json").read_text())
    if selected_schema != CUB_USED_ATTRIBUTE_INDICES:
        raise RuntimeError(
            f"{args.pkls}/selection_indices.json does not match the canonical "
            "zero-based 112-index schema; rerun data/cub70/prepare_all.sh"
        )

    splits = {}
    for split in ("train", "test"):
        with (args.pkls / f"{split}.pkl").open("rb") as handle:
            records = pickle.load(handle)
        widths = {len(record["attribute_label"]) for record in records}
        if widths != {112}:
            raise RuntimeError(f"{split}: expected 112 labels, got {sorted(widths)}")
        splits[split] = records

    names_by_id = load_numbered_names(args.attr_dir / "attributes.txt")
    missing_ids = [idx for idx in CUB_USED_ATTRIBUTE_IDS if idx not in names_by_id]
    if missing_ids:
        raise RuntimeError(f"attributes.txt missing canonical IDs: {missing_ids}")
    selected_names = [names_by_id[idx] for idx in CUB_USED_ATTRIBUTE_IDS]
    groups = sorted({name.split("::", 1)[0] for name in selected_names})
    if len(groups) != 28:
        raise RuntimeError(f"canonical 112 attributes unexpectedly span {len(groups)} groups")

    text = args.config.read_text()
    match = re.search(r"^\s*n_groups_concepts:\s*(\d+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"{args.config}: no n_groups_concepts")
    configured_groups = int(match.group(1))
    if configured_groups != len(groups):
        raise RuntimeError(
            f"{args.config}: requests {configured_groups} groups, but inputs contain "
            f"{len(groups)} ({groups})"
        )

    classes = sorted({int(record["class_label"]) for record in splits["train"]})
    expected_classes = 200 if args.dataset == "cub" else 70
    if classes != list(range(expected_classes)):
        raise RuntimeError(
            f"{args.dataset}: expected classes 0..{expected_classes - 1}, got "
            f"{classes[:3]}..{classes[-3:]} ({len(classes)} total)"
        )

    positives = [
        sum(int(record["attribute_label"][j]) for record in splits["train"])
        for j in range(112)
    ]
    zero_positive = [j for j, count in enumerate(positives) if count == 0]
    all_positive = [
        j for j, count in enumerate(positives) if count == len(splits["train"])
    ]
    if args.dataset == "cub" and zero_positive:
        raise RuntimeError(
            f"full CUB has all-zero canonical concepts {zero_positive}; "
            "this indicates an attribute-indexing error"
        )
    if zero_positive or all_positive:
        print(
            f"[CUB preflight warning] constant training concepts: "
            f"all-zero={zero_positive}, all-one={all_positive}. "
            "They are retained for the shared 112-concept schema; the loader "
            "uses neutral positive weight only for all-zero targets; targets "
            "with positives retain Koh's exact imbalance formula."
        )

    print(
        f"[CUB preflight OK] dataset={args.dataset} "
        f"train={len(splits['train'])} test={len(splits['test'])} "
        f"classes={len(classes)} concepts=112 groups=28"
    )


if __name__ == "__main__":
    main()
