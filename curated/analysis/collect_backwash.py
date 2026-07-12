#!/usr/bin/env python3
"""Collect grounding parquets -> BACKWASH-vs-gamma table + the money figure.

Reads $CURATED_DATA/grounding/*.parquet (one per model, written by
grounding_deletion.py, named funnybirds-<model>[-g<tag>]-s<seed>.parquet) and
produces:
  <out>.csv   one row per (model, gamma, seed): p_intact, p_removed, retained_frac
  <out>.png/.pdf   retained_frac vs gamma; MCBM curve + CBM/vanilla reference lines.

retained_frac = p_removed / p_intact = fraction of a concept's prob that survives
deleting its part. Read as backwash: ~1 = the model still 'sees' a part that isn't
there (species-lookup); ~0 = grounded (concept collapses when the part is gone).
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
        # retained_frac = P(typical concept | part removed) / P(concept | intact).
        # Raw operational metric; read as backwash where consumed (see notebook 03).
        retained = prm / pi if pi > 1e-6 else np.nan
        rows.append(dict(model=model, gamma=gamma, seed=seed, n=len(df),
                         p_intact=pi, p_removed=prm, retained_frac=retained))
    if not rows:
        print(f"[collect] no grounding parquets in {args.grounding}")
        return
    T = pd.DataFrame(rows).sort_values(["model", "gamma", "seed"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(args.out + ".csv", index=False)   # full table incl. any diverged rows
    print(T.to_string(index=False))

    bad = T[T.retained_frac.isna()]
    if len(bad):
        print("\n[collect] EXCLUDED from figure (diverged/NaN -> retrain): "
              + ", ".join(f"{r.model} g{r.gamma}" for r in bad.itertuples()))
    T = T[T.retained_frac.notna()]                # clean rows only for plotting

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    mc = T[T.model == "mcbm"]
    if len(mc):
        g = mc.groupby("gamma").retained_frac.agg(["mean", "std"]).reset_index()
        pos = g.gamma[g.gamma > 0]
        floor = (pos.min() / 3) if len(pos) else 0.01
        x = g.gamma.replace(0, floor)                 # place gamma=0 on the log axis
        ax.errorbar(x, g["mean"], yerr=g["std"].fillna(0), marker="o",
                    capsize=3, label="MCBM (γ sweep)")
        ax.set_xscale("log")
    for model, color in (("cbm", "tab:orange"), ("vanilla", "tab:gray")):
        sub = T[T.model == model]
        if len(sub):
            ax.axhline(sub.retained_frac.mean(), ls="--", color=color, label=f"{model} (ref)")
    ax.set_xlabel("γ  (minimality; effective force = γ × 0.2)")
    ax.set_ylabel("retained_frac of a REMOVED part\n(P concept removed / P concept intact)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Removed-part concept retention vs bottleneck strength\n(FunnyBirds, deletion test)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=140)
    plt.savefig(args.out + ".pdf")
    print(f"[collect] wrote {args.out}.csv/.png/.pdf")


if __name__ == "__main__":
    main()
