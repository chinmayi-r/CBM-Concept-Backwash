#!/usr/bin/env python3
"""Step 1 — FunnyBirds DATA analysis (dataset only, no model, CPU, seconds).

Characterizes the built dataset and answers the questions that decide which
downstream analyses are valid (mirrors notebooks/01, headless for the cluster):
  1. class balance
  2. concept prevalence (per one-hot concept, grouped by part)
  3. ground-truth class x concept matrix
  4. SPECIES-CONSTANCY — within-species std of each concept on test. If ~0, the
     matched-pair recall gap has no within-species signal on FunnyBirds (n=10
     quantization) -> recall gap is a CUB tool; FunnyBirds uses deletion/swap.
  5. absent parts (all-zero concept groups)

Reads $CURATED_DATA/funnybirds_processed/{train,test}.pkl + parts.json.
Writes figures + a summary to <out> (default $CURATED_DATA/data_analysis).

  python analysis/data_analysis.py \
    --funnybirds-root $CURATED_DATA/FunnyBirds \
    --pkls $CURATED_DATA/funnybirds_processed \
    --out $CURATED_DATA/data_analysis
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data" / "funnybirds"))


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds-root", required=True)
    ap.add_argument("--pkls", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import funnybirds_concepts as fbc
    parts = fbc.load_parts(args.funnybirds_root)
    names = fbc.concept_names(parts)
    spans = fbc.group_slices(parts)
    nC = len(names)
    part_of = {j: p for p, (a, b) in spans.items() for j in range(a, b)}
    colors = {p: c for p, c in zip(spans, plt.cm.tab10.colors)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr = _load(Path(args.pkls) / "train.pkl")
    te = _load(Path(args.pkls) / "test.pkl")
    Atr = np.array([r["attribute_label"] for r in tr]); ytr = np.array([r["class_label"] for r in tr])
    Ate = np.array([r["attribute_label"] for r in te]); yte = np.array([r["class_label"] for r in te])
    assert Atr.shape[1] == nC, f"concept width {Atr.shape[1]} != {nC}"
    summary = []
    summary.append(f"train {len(tr)} imgs, {len(set(ytr))} classes; test {len(te)} imgs, {len(set(yte))} classes; {nC} concepts")
    print(summary[-1])

    # 1. class balance
    trc = pd.Series(ytr).value_counts(); tec = pd.Series(yte).value_counts()
    summary.append(f"train/class: min={trc.min()} max={trc.max()} (equal={trc.min()==trc.max()}); "
                   f"test/class: min={tec.min()} max={tec.max()}")
    print(summary[-1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    ax[0].bar(trc.sort_index().index, trc.sort_index().values); ax[0].set_title("train imgs/species")
    ax[1].bar(tec.sort_index().index, tec.sort_index().values, color="tab:orange"); ax[1].set_title("test imgs/species")
    plt.tight_layout(); plt.savefig(out / "01_class_balance.png", dpi=120); plt.close()

    # 2. concept prevalence
    prev = Atr.mean(0)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.bar(range(nC), prev, color=[colors[part_of[j]] for j in range(nC)])
    ax.set_xticks(range(nC)); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("P(concept=1) train"); ax.set_title("Concept prevalence (by part)")
    plt.tight_layout(); plt.savefig(out / "02_concept_prevalence.png", dpi=120); plt.close()
    summary.append(f"rarest concepts: {[names[i] for i in np.argsort(prev)[:5]]} (min prev {prev.min():.3f})")
    print(summary[-1])

    # 3. class x concept matrix
    M = pd.DataFrame(Atr).assign(c=ytr).groupby("c").mean().values
    frac_binary = float(np.mean((M == 0) | (M == 1)))
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(nC)); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("species"); ax.set_title("class x concept (train mean)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout(); plt.savefig(out / "03_class_concept_matrix.png", dpi=120); plt.close()
    summary.append(f"class x concept cells exactly 0/1: {frac_binary:.4f} "
                   f"({'clean species-level lookup' if frac_binary > 0.999 else 'within-species variation present'})")
    print(summary[-1])

    # 4. species-constancy (the decision-driver)
    within = np.array([Ate[yte == c].std(0) for c in np.unique(yte)])
    frac_const = float(np.mean(within == 0))
    n_img = int(min((yte == c).sum() for c in np.unique(yte)))
    summary.append(f"SPECIES-CONSTANCY: (species,concept) pairs with within-species std==0: {frac_const:.4f}; "
                   f"mean within-species std {within.mean():.4g}; test imgs/species={n_img}")
    print(summary[-1])
    if frac_const > 0.999:
        v = (f"VERDICT: concepts are species-constant on test -> matched-pair recall gap is n={n_img} "
             f"quantization noise on FunnyBirds. Use deletion/swap here; recall gap is the CUB axis.")
    else:
        v = "VERDICT: within-species variation exists -> recall gap may be testable (check n & significance)."
    summary.append(v); print(v)

    # 5. absent parts
    rows = []
    for p, (a, b) in spans.items():
        grp = Atr[:, a:b].sum(1)
        rows.append((p, b - a, int((grp == 0).sum()), float((grp == 0).mean())))
    absdf = pd.DataFrame(rows, columns=["part", "n_variants", "absent_imgs", "absent_frac"])
    summary.append("absent parts (all-zero group):\n" + absdf.to_string(index=False))
    print(summary[-1])

    # 6. PART DIFFICULTY PROFILE — the dataset-side predictors of which part backwashes
    #    (matches the old renderer-swap finding: variant count predicts it; occlusion
    #     does NOT). tail has the most variants (9) and is the backwash-prone part.
    prof = []
    vis_path = Path(args.pkls).parent / "funnybirds_visibility.parquet"
    vis = pd.read_parquet(vis_path) if vis_path.exists() else None
    for p, (a, b) in spans.items():
        row = {"part": p, "n_variants": b - a}
        if vis is not None:
            vp = vis[vis.part == p]
            if len(vp):
                row["mean_px"] = round(float(vp.pixel_count.mean()), 1)
                row["median_px"] = float(vp.pixel_count.median())
                row["frac_visible"] = round(float((vp.pixel_count > 0).mean()), 3)
        prof.append(row)
    profdf = pd.DataFrame(prof).sort_values("n_variants", ascending=False)
    summary.append("\n=== PART DIFFICULTY PROFILE (dataset predictors of backwash) ===")
    summary.append(profdf.to_string(index=False))
    note = ("Interpretation: VARIANT COUNT is the structural predictor — tail has the most "
            "(9) -> largest/hardest concept space -> model falls back on the species prior "
            "(the 0.36 tail backwash). PIXEL AREA / visibility is the OCCLUSION control: the "
            "renderer-swap analysis showed frac_correct is FLAT across pixel quartiles, so "
            "occlusion does NOT explain tail failure — it is genuine backwash (species anchoring).")
    summary.append(note); print("\n" + profdf.to_string(index=False) + "\n" + note)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    order = profdf.part.tolist()
    ax.bar(order, profdf.n_variants, color=[colors[p] for p in order])
    ax.set_ylabel("# variants"); ax.set_title("Part difficulty: variant count (tail=9 -> most backwash-prone)")
    for i, v in enumerate(profdf.n_variants):
        ax.text(i, v + 0.1, str(int(v)), ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(out / "06_part_difficulty.png", dpi=120); plt.close()

    (out / "SUMMARY.txt").write_text("\n".join(str(s) for s in summary))
    print(f"\n[data_analysis] figures + SUMMARY.txt -> {out}")


if __name__ == "__main__":
    main()
