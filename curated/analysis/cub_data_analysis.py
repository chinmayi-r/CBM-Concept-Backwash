#!/usr/bin/env python3
"""CUB + CUB70 DATA analysis (dataset only, no model, CPU, seconds).

The CUB analog of analysis/data_analysis.py. Runs on the pickles that already
exist ($CURATED_DATA/CUB_processed/...); NO masks needed. Answers the question
that decides whether the recall-gap axis is usable on CUB:

  SPECIES-CONSTANCY: within-species std of each attribute on test.
    ~0  -> concepts are class-level (like FunnyBirds) -> recall gap underpowered.
    >0  -> attributes VARY within a species -> matched-pair recall gap has signal
           (this is the CUB axis FunnyBirds can't provide).

Also: class balance, attribute prevalence, class x attribute binary fraction.
Compares FULL CUB (class_attr_data_10) vs CUB70 (class_attr_data_10_cub70_original).

  python analysis/cub_data_analysis.py            # both datasets
  python analysis/cub_data_analysis.py --out $CURATED_DATA/cub_data_analysis
"""
from __future__ import annotations
import argparse, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = {
    "cub":   "class_attr_data_10",
    "cub70": "class_attr_data_10_cub70_original",
}


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def analyse(name, pkls_dir, lines):
    tr_p, te_p = pkls_dir / "train.pkl", pkls_dir / "test.pkl"
    if not te_p.exists():
        lines.append(f"[{name}] missing {te_p} -> skip"); return
    tr = _load(tr_p) if tr_p.exists() else []
    te = _load(te_p)
    Ate = np.array([r["attribute_label"] for r in te])
    yte = np.array([r["class_label"] for r in te])
    nC = Ate.shape[1]
    lines.append(f"\n{'='*66}\n### {name}  ({pkls_dir.name})\n{'='*66}")
    lines.append(f"train {len(tr)} / test {len(te)} imgs · {len(set(yte))} species · {nC} attributes")

    # class balance (test)
    ec = pd.Series(yte).value_counts()
    lines.append(f"test imgs/species: min={ec.min()} max={ec.max()} (median {int(ec.median())})")

    # attribute prevalence
    prev = Ate.mean(0)
    lines.append(f"attribute prevalence: min {prev.min():.3f}  max {prev.max():.3f}  mean {prev.mean():.3f}")

    # class x attribute binary fraction (on test means)
    M = pd.DataFrame(Ate).assign(c=yte).groupby("c").mean().values
    frac_bin = float(np.mean((M == 0) | (M == 1)))
    lines.append(f"class x attribute cells exactly 0/1 (test): {frac_bin:.4f} "
                 f"({'class-level labels' if frac_bin > 0.999 else 'within-species variation present'})")

    # SPECIES-CONSTANCY — the decision driver
    within = np.array([Ate[yte == c].std(0) for c in np.unique(yte)])
    frac_const = float(np.mean(within == 0))
    lines.append(f"SPECIES-CONSTANCY: (species,attr) pairs with within-species std==0: {frac_const:.4f}; "
                 f"mean within-species std {within.mean():.4g}")
    if frac_const > 0.999:
        v = ("VERDICT: concepts are species-constant on test -> recall gap underpowered here "
             "(like FunnyBirds); rely on deletion/occlusion.")
    else:
        pct = (1 - frac_const) * 100
        v = (f"VERDICT: {pct:.1f}% of (species,attr) pairs VARY within a species -> matched-pair "
             f"recall gap HAS signal on {name} (the CUB axis FunnyBirds lacks).")
    lines.append(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.data_root) / "CUB_processed"

    lines = []
    for name, sub in DATASETS.items():
        d = root / sub
        if d.exists():
            analyse(name, d, lines)
        else:
            lines.append(f"[{name}] {d} not found -> skip")

    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "SUMMARY.txt").write_text(text)
        print(f"\n[cub_data_analysis] wrote {out/'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
