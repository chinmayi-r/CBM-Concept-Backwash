#!/usr/bin/env python3
"""Collect grounding parquets -> BACKWASH-vs-gamma table + the money figure.

Reads $CURATED_DATA/grounding/*.parquet (one per model, written by
grounding_deletion.py, named funnybirds-<model>[-g<tag>]-s<seed>.parquet) and
produces:
  <out>.csv   one row per (model, gamma, seed): p_intact, p_removed, backwash
  <out>.png/.pdf   backwash vs gamma; MCBM curve + CBM/vanilla reference lines.

backwash = 1 - (p_intact - p_removed)/p_intact  = retained prob of a REMOVED part.
1 = the model fully 'sees' parts that aren't there (species-lookup); 0 = grounded.
"""
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME = re.compile(r"^funnybirds-(vanilla|cbm|mcbm)(?:-g([0-9p]+))?-s(\d+)$")


def parse(stem):
    m = NAME.match(stem)
    if not m:
        return None
    model, gtag, seed = m.group(1), m.group(2), int(m.group(3))
    gamma = float(gtag.replace("p", ".")) if gtag else np.nan
    return model, gamma, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounding", required=True)
    ap.add_argument("--out", required=True, help="output path stem")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.grounding, "*.parquet"))):
        pr = parse(Path(f).stem)
        if pr is None:
            continue
        model, gamma, seed = pr
        df = pd.read_parquet(f)
        pi, prm = float(df.p_intact.mean()), float(df.p_removed.mean())
        grounding = (pi - prm) / pi if pi > 1e-6 else np.nan
        rows.append(dict(model=model, gamma=gamma, seed=seed, n=len(df),
                         p_intact=pi, p_removed=prm,
                         grounding=grounding, backwash=1 - grounding))
    if not rows:
        print(f"[collect] no grounding parquets in {args.grounding}")
        return
    T = pd.DataFrame(rows).sort_values(["model", "gamma", "seed"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(args.out + ".csv", index=False)
    print(T.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    mc = T[T.model == "mcbm"]
    if len(mc):
        g = mc.groupby("gamma").backwash.agg(["mean", "std"]).reset_index()
        pos = g.gamma[g.gamma > 0]
        floor = (pos.min() / 3) if len(pos) else 0.01
        x = g.gamma.replace(0, floor)                 # place gamma=0 on the log axis
        ax.errorbar(x, g["mean"], yerr=g["std"].fillna(0), marker="o",
                    capsize=3, label="MCBM (γ sweep)")
        ax.set_xscale("log")
    for model, color in (("cbm", "tab:orange"), ("vanilla", "tab:gray")):
        sub = T[T.model == model]
        if len(sub):
            ax.axhline(sub.backwash.mean(), ls="--", color=color, label=f"{model} (ref)")
    ax.set_xlabel("γ  (minimality; effective force = γ × 0.2)")
    ax.set_ylabel("backwash  (retained P of a REMOVED part)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Concept–class backwash vs bottleneck strength\n(FunnyBirds, deletion test)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=140)
    plt.savefig(args.out + ".pdf")
    print(f"[collect] wrote {args.out}.csv/.png/.pdf")


if __name__ == "__main__":
    main()
