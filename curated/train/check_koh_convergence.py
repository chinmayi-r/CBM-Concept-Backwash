#!/usr/bin/env python3
"""Summarize progress toward Koh's built-in 100-epoch early stop.

This reads Slurm output only.  It never changes or cancels a job.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch \[(\d+)\].*Best val epoch: (-?\d+)")


def summarize(path: Path) -> tuple[int, int] | None:
    latest = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if match:
                latest = (int(match.group(1)), int(match.group(2)))
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    print(
        f"{'log':55} {'epoch':>7} {'best':>7} {'since_best':>11} "
        f"{'to_early_stop':>13}"
    )
    print("-" * 99)
    for path in args.logs:
        if not path.is_file():
            print(f"{str(path):55} {'MISSING':>40}")
            continue
        result = summarize(path)
        if result is None:
            print(f"{path.name:55} {'NO EPOCH YET':>40}")
            continue
        epoch, best = result
        since_best = epoch - best
        # train.py checks `epoch - best_val_epoch >= 100` after printing the
        # epoch, so zero means that its own early-stop condition is now due.
        remaining = max(0, 100 - since_best)
        print(
            f"{path.name:55} {epoch:7d} {best:7d} {since_best:11d} "
            f"{remaining:13d}"
        )


if __name__ == "__main__":
    main()
