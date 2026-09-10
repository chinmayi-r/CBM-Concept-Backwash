#!/usr/bin/env python3
"""Synthetic contract checks for strict RGB and configurable part-map validation."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import pandas as pd

from validate_fixed_swaps import main


def run_validator(root: Path, policy: str) -> str:
    previous = sys.argv
    stream = io.StringIO()
    try:
        sys.argv = ["validate_fixed_swaps.py", "--out", str(root), "--part-map-policy", policy]
        with contextlib.redirect_stdout(stream):
            main()
    finally:
        sys.argv = previous
    return stream.getvalue()


def fixture(root: Path) -> None:
    rows = []
    for index in range(10):
        rows.append(dict(
            li=index,
            render_id=f"cf-{index}",
            image_cf_sha256=f"cf-sha-{index}",
            partmap_cf_sha256=f"map-sha-{index}",
            orig_render_id=f"orig-{index % 5}",
            image_orig_sha256=f"orig-sha-{index % 5}",
            part="tail",
            direction="fwd",
            sid_src=0,
            sid_donor=1,
            var_src=0,
            var_donor=1,
            pixel_count_cf=10,
        ))
    first = pd.DataFrame(rows)
    second = first.copy()
    second.loc[0, "partmap_cf_sha256"] = "historical-byte-hash-difference"
    first.to_csv(root / "model-a-s1.csv", index=False)
    second.to_csv(root / "model-b-s1.csv", index=False)


def test_disclose_allows_only_auxiliary_partmap_mismatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture(root)
        output = run_validator(root, "disclose")
        assert "AUXILIARY PART-MAP METADATA DISCLOSURE" in output
        assert "part_map_mismatches=1" in output

        try:
            run_validator(root, "strict")
        except RuntimeError as error:
            assert "part_map=1" in str(error)
        else:
            raise AssertionError("strict mode accepted a part-map mismatch")

        path = root / "model-b-s1.csv"
        changed = pd.read_csv(path)
        changed.loc[0, "image_cf_sha256"] = "different-rgb-input"
        changed.to_csv(path, index=False)
        try:
            run_validator(root, "disclose")
        except RuntimeError as error:
            assert "counterfactual_rgb=1" in str(error)
        else:
            raise AssertionError("disclose mode accepted an RGB model-input mismatch")


if __name__ == "__main__":
    test_disclose_allows_only_auxiliary_partmap_mismatch()
    print("FIXED SWAP VALIDATOR SYNTHETIC PASS")
