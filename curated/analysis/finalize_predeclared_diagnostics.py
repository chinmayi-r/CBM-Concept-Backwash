"""Write an atomic provenance manifest after every D6 table exists."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import diag_common as dc

EXPECTED = [
    "conflict_exact.csv",
    "d61_dimension_adjusted_information.csv",
    "d61_tail_subset_repeats.csv",
    "d61_tail_subset_sensitivity.csv",
    "d62_profile_transfer_rows.csv",
    "d62_profile_transfer_summary.csv",
    "d63_conflict_value_level.csv",
    "d63_conflict_row_level_ols.csv",
    "d64_risk_model_heldout.csv",
    "d64_event_model_heldout.csv",
    "d64_risk_model_coefficients.csv",
    "d64_risk_model_transport.csv",
    "d65_saved_head_use.csv",
]
SCRIPTS = [
    "diag_common.py",
    "diag_dimension_adjusted_information.py",
    "diag_profile_transfer.py",
    "diag_conflict_components.py",
    "diag_grouped_risk_model.py",
    "diag_saved_head_use.py",
    "finalize_predeclared_diagnostics.py",
]


def describe(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty required file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": dc.sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--swap-root", required=True, type=Path)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    inputs = [
        args.model_root / "SUCCESS.json",
        args.model_root / "final_test.parquet",
        args.model_root / "final_model_1.pth",
        args.swap_root / "SUCCESS.json",
        args.swap_root / "funnybirds-cbm-s1.csv",
    ]
    curated = dc.curated_root()
    for labels in ("standard", "rlv2"):
        for split in ("train", "val"):
            inputs.append(curated / "koh_joint_inputs" / "funnybirds" /
                          labels / f"{split}.pkl")
    funnybirds = Path(os.environ.get("FUNNYBIRDS_ROOT", curated / "FunnyBirds"))
    inputs.append(funnybirds / "parts.json")
    payload = {
        "status": "SUCCESS",
        "claim_scope": "read-only FunnyBird Standard-CBM seed-1 diagnostics",
        "framework": "Koh Joint ResNet-50",
        "training_performed": False,
        "rendering_performed": False,
        "git_commit": commit,
        "inputs": [describe(path) for path in inputs],
        "scripts": [describe(here / name) for name in SCRIPTS],
        "outputs": [describe(args.output / name) for name in EXPECTED],
        "uncertainty_scope": (
            "original-image resampling for this fixed seed-1 model; "
            "not training-seed uncertainty"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "SUCCESS.json"
    fd, temporary = tempfile.mkstemp(
        prefix="SUCCESS.", suffix=".tmp", dir=args.output)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"[D6 SUCCESS MANIFEST] {destination}")


if __name__ == "__main__":
    main()
