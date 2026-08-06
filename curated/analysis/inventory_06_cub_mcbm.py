#!/usr/bin/env python3
"""Inventory existing CUB/CUB70 MCBM artifacts without training or Slurm."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch


PATTERN = re.compile(r"^(cub70|cub)-mcbm-g([0-9p.]+)$")


def epoch(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def finite_prediction(path: Path) -> tuple[bool, str]:
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        required = ["z", "c", "c_preds", "y", "y_preds"]
        missing = [key for key in required if key not in d]
        if missing:
            return False, f"missing keys {missing}"
        bad = [key for key in required if not torch.isfinite(d[key]).all()]
        if bad:
            return False, f"non-finite tensors {bad}"
        return True, "finite required tensors"
    except Exception as exc:
        return False, f"load error: {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated-data", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    curated = Path(__file__).resolve().parent.parent
    results = curated / "external/minimal_cbm/results"
    eval_root = Path(args.curated_data) / "cub70_eval"
    rows = []

    if results.exists():
        for config in sorted(results.iterdir()):
            match = PATTERN.fullmatch(config.name)
            if not match or not config.is_dir():
                continue
            dataset, gamma_text = match.groups()
            gamma = float(gamma_text.replace("p", "."))
            for seed_dir in sorted(config.glob("[0-9]*")):
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                models = {epoch(p): p for p in (seed_dir / "models").glob("epoch_*.pt")}
                predictions = {
                    epoch(p): p for p in (seed_dir / "predictions").glob("epoch_*.pth")
                }
                common = sorted(set(models) & set(predictions))
                if not common:
                    rows.append({
                        "dataset": dataset, "config": config.name,
                        "gamma": gamma, "seed": int(seed_dir.name),
                        "status": "INCOMPLETE", "latest_common_epoch": None,
                        "detail": "no matching model/prediction epoch",
                        "export": "",
                    })
                    continue
                selected = 100 if 100 in common else common[-1]
                valid, detail = finite_prediction(predictions[selected])
                export = eval_root / f"{config.name}-s{seed_dir.name}.parquet"
                rows.append({
                    "dataset": dataset, "config": config.name,
                    "gamma": gamma, "seed": int(seed_dir.name),
                    "status": "ACCEPTED FOR EXPORT" if valid else "INVALID OUTPUT",
                    "latest_common_epoch": selected,
                    "detail": detail,
                    "export": str(export) if export.exists() else "",
                })

    table = pd.DataFrame(rows)
    if table.empty:
        print("INCOMPLETE: no cub/cub70 MCBM result directories found")
    else:
        table = table.sort_values(["dataset", "gamma", "seed"]).reset_index(drop=True)
        print(table.to_string(index=False))
        accepted = table[table.status == "ACCEPTED FOR EXPORT"]
        print("\nAccepted finite checkpoint/prediction pairs:")
        print(accepted.groupby(["dataset", "gamma"]).seed.apply(list).to_string())
        print("\nMissing normalized exports:")
        missing = accepted[accepted.export == ""]
        if missing.empty:
            print("none")
        else:
            for row in missing.itertuples():
                print(
                    "python analysis/cub70_export_eval.py "
                    f"--config {row.config} --seed {row.seed} "
                    f"--epoch {row.latest_common_epoch} "
                    f"--out \"$CURATED_DATA/cub70_eval/{row.config}-s{row.seed}.parquet\""
                )

    out = Path(args.out) if args.out else eval_root / "mcbm_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    audit = {
        "status": "ACCEPTED FOR ARTIFACT INVENTORY",
        "rows": len(table),
        "accepted": int((table.status == "ACCEPTED FOR EXPORT").sum()) if len(table) else 0,
        "invalid": int((table.status == "INVALID OUTPUT").sum()) if len(table) else 0,
        "incomplete": int((table.status == "INCOMPLETE").sum()) if len(table) else 0,
    }
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
