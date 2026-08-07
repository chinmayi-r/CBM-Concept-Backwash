#!/usr/bin/env python3
"""Enumerate the canonical matrix and require every stage manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def expected() -> list[str]:
    names: list[str] = []
    koh_cells = [("funnybirds", "standard"), ("funnybirds", "rlv2"),
                 ("cub", "standard"), ("cub70", "standard")]
    for seed in (1, 2, 3):
        for dataset, labels in koh_cells:
            for stage in ("koh_concept", "koh_independent", "koh_extract",
                          "koh_sequential", "koh_joint", "koh_joint_sigmoid"):
                names.append(f"{stage}_{dataset}_{labels}_s{seed}")
            for variant in ("independent", "sequential", "joint", "joint_sigmoid"):
                names.append(f"eval_koh_{dataset}_{labels}_{variant}_s{seed}")
        for dataset, labels in koh_cells:
            names.append(f"minimal_cbm_cbm_{dataset}_{labels}_s{seed}")
            names.append(f"eval_minimal_cbm_cbm_{dataset}_{labels}_s{seed}")
            for gamma in ("0", "0p1", "0p3", "1", "3", "5"):
                names.append(f"mcbm_{dataset}_{labels}_g{gamma}_s{seed}")
                names.append(f"eval_mcbm_{dataset}_{labels}_g{gamma}_s{seed}")
        names.append(f"swap_all_funnybirds_standard_s{seed}")
        names.append(f"swap_all_funnybirds_rlv2_s{seed}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    manifest_dir = args.root / "manifests"
    rows, missing, invalid = [], [], []
    for name in expected():
        path = manifest_dir / f"{name}.json"
        status = "MISSING"
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                status = data.get("status", "INVALID")
            except Exception:
                status = "INVALID"
        if status == "MISSING": missing.append(name)
        elif status != "SUCCESS": invalid.append(name)
        rows.append({"stage": name, "status": status})
    out = args.root / "completion.json"
    result = {
        "expected_stage_manifests": len(rows),
        "successful": sum(row["status"] == "SUCCESS" for row in rows),
        "missing": missing, "invalid": invalid, "rows": rows,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[CANONICAL COMPLETION] {result['successful']}/{len(rows)} -> {out}")
    if missing or invalid:
        raise SystemExit(f"INCOMPLETE: missing={len(missing)} invalid={len(invalid)}")


if __name__ == "__main__":
    main()
