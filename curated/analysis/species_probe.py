#!/usr/bin/env python3
"""Species-identity probe on the bottleneck — is the concept layer a class code?

Ports fb_cbm_counterfactual.ipynb §6 onto the official minimal_cbm models,
renderer-free (plain test images only). The backwash mechanism, stated as a
measurement: if a part concept z_j reads *species* rather than *its part*, then
species identity is linearly recoverable from the bottleneck. We fit a simple
linear probe species <- z (and species <- c_preds) and compare to chance (1/50):

  z (continuous bottleneck)  : upper bound on how much class code the rep carries.
  c_preds (concept probs)    : the 26-d concept vector the paper exposes. High
                               here means the *reported concepts alone* pin the
                               species -- concept=f(class) is realized, so a
                               part concept can be answered by class-lookup
                               without ever reading its part (that is backwash).

This does not by itself prove any single concept is backwashed (the deletion
test in grounding_deletion.py is the causal probe). It quantifies the standing
opportunity: the degree to which the bottleneck is a species code. Read the two
together -- high species-from-c AND per-part deletion-backwash = the mechanism.

  python analysis/species_probe.py \
      --config funnybirds-cbm --seed 1 \
      --funnybirds-root $CURATED_DATA/FunnyBirds \
      --pkls $CURATED_DATA/funnybirds_processed \
      --out $CURATED_DATA/species_probe/funnybirds-cbm-seed1.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

HERE = Path(__file__).resolve().parent            # curated/analysis
CURATED = HERE.parent                             # curated
MCBM = CURATED / "external" / "minimal_cbm"
COMPAT = CURATED / "compat"
FBDATA = CURATED / "data" / "funnybirds"
for p in (str(MCBM), str(COMPAT), str(FBDATA)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

# reuse the exact model loader so the architecture/weights match training
from grounding_deletion import load_model, _MEAN, _STD  # noqa: E402


@torch.inference_mode()
def _bottleneck(model, imgs: torch.Tensor, n_concepts: int, device):
    """(B,3,H,W) -> (z (B,dz), c_preds (B,n_concepts))."""
    c_dummy = torch.zeros(imgs.shape[0], n_concepts, device=device)
    out = model(imgs.to(device), c_dummy)
    z = out["z"].detach().cpu().numpy()
    z = z.reshape(z.shape[0], -1)                 # flatten any per-concept dims
    c = out["c_preds"][..., 0].detach().cpu().numpy()
    return z, c


def _cv_probe(X, y, seed):
    """5-fold CV accuracy of a linear (logistic) probe, plus chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(acc.mean()), float(acc.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--funnybirds-root", required=True)
    ap.add_argument("--pkls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N test images (0=all)")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    import funnybirds_concepts as fbc
    fb = Path(args.funnybirds_root)
    parts = fbc.load_parts(fb)
    spans = fbc.group_slices(parts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_concepts = load_model(args.config, args.seed, args.epoch, device)

    tfm = T.Compose([
        T.Resize(int(args.img_size * 256 / 224)),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])

    params = json.loads((fb / "dataset_test.json").read_text())
    if args.limit:
        params = params[: args.limit]

    Zs, Cs, ys = [], [], []
    batch, meta = [], []
    BS = 64

    def flush():
        if not batch:
            return
        z, c = _bottleneck(model, torch.stack(batch), n_concepts, device)
        Zs.append(z); Cs.append(c); ys.extend(meta)
        batch.clear(); meta.clear()

    n_missing = 0
    for idx, entry in enumerate(params):
        c = int(entry["class_idx"])
        stem = f"{idx:06d}"
        path = fb / "test" / str(c) / f"{stem}.png"
        if not path.exists():
            n_missing += 1
            continue
        batch.append(tfm(Image.open(path).convert("RGB")))
        meta.append(c)
        if len(batch) >= BS:
            flush()
    flush()

    Z = np.concatenate(Zs); C = np.concatenate(Cs); y = np.asarray(ys)
    n_species = len(np.unique(y))
    chance = 1.0 / n_species
    print(f"[species_probe] {len(y)} imgs, {n_species} species, {n_missing} missing; "
          f"z dim {Z.shape[1]}, c dim {C.shape[1]}, chance {chance:.3f}")

    z_acc, z_sd = _cv_probe(Z, y, args.seed)
    c_acc, c_sd = _cv_probe(C, y, args.seed)

    # per-part: how much species is recoverable from EACH part's concept block
    # alone (a part whose concepts alone pin the species is doing class-coding).
    per_part = {}
    for p, (a, b) in spans.items():
        try:
            pa, ps = _cv_probe(C[:, a:b], y, args.seed)
            per_part[p] = {"acc": round(pa, 4), "sd": round(ps, 4), "n_variants": b - a}
        except Exception as e:  # a degenerate (all-constant) block
            per_part[p] = {"error": str(e), "n_variants": b - a}

    result = {
        "config": args.config, "seed": args.seed,
        "n_imgs": int(len(y)), "n_species": int(n_species), "chance": chance,
        "species_from_z":      {"acc": round(z_acc, 4), "sd": round(z_sd, 4)},
        "species_from_cpreds": {"acc": round(c_acc, 4), "sd": round(c_sd, 4)},
        "species_from_part_cpreds": per_part,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print("\n=== species recoverability (5-fold CV accuracy; chance = %.3f) ===" % chance)
    print(f"  from z (bottleneck) : {z_acc:.3f} ± {z_sd:.3f}")
    print(f"  from c_preds (26-d) : {c_acc:.3f} ± {c_sd:.3f}   "
          f"<- how much the REPORTED concepts alone pin the species")
    print("  from each part's concept block alone:")
    for p, d in sorted(per_part.items(), key=lambda kv: -kv[1].get("acc", 0)):
        if "acc" in d:
            print(f"    {p:6s} ({d['n_variants']} variants): {d['acc']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
