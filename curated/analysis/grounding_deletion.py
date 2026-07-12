#!/usr/bin/env python3
"""Deletion grounding test — does a trained CBM/MCBM still 'see' a removed part?

The clean, off-manifold backwash probe (DECISIONS D.3). For each FunnyBirds test
image and each part p that the species actually HAS, we load the pre-rendered
intervention image with p removed
    test_interventions/<class_idx>/<image_idx:06d>/body_<sorted kept parts>.png
run the trained model, and read the predicted probability of the SPECIES-TYPICAL
concept for part p (the variant the species canonically has):

  grounded   -> that probability DROPS when p is removed (it read the pixels).
  backwashed -> that probability STAYS high (it inferred p from the species /
                the rest of the bird, reporting a part it cannot see).

Per part we report:
  p_intact   mean prob of the typical concept on the full bird   (~ concept acc)
  p_removed  mean prob of the SAME concept with the part removed
  drop        = p_intact - p_removed
  grounding   = drop / p_intact   in [0,1]   (1 = fully grounded, 0 = full backwash)
  backwash    = 1 - grounding                (headline number)

No manifold bookkeeping: removing a part can't drift you to another species, so a
confident concept on a part-less bird is unambiguous backwash.

Runs on the cluster where CURATED_DATA + the trained checkpoint exist:
  python analysis/grounding_deletion.py \
      --config funnybirds-cbm --seed 1 \
      --funnybirds-root $CURATED_DATA/FunnyBirds \
      --pkls $CURATED_DATA/funnybirds_processed \
      --out $CURATED_DATA/grounding/funnybirds-cbm-seed1.parquet
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

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _latest_epoch_ckpt(model_dir: Path) -> Path:
    ck = sorted(model_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ck:
        raise FileNotFoundError(f"no epoch_*.pt in {model_dir}")
    return ck[-1]


def load_model(config_basename: str, seed: int, epoch: int | None, device):
    """Rebuild the exact minimal_cbm model and load the trained weights."""
    from src.helpers import read_config
    from src.models import get_model
    from mcbm_funnybirds import get_funnybirds

    prefix = config_basename.split("-")[0]
    cfg = read_config(str(MCBM / "configs" / prefix / config_basename))
    # model_kwargs (n_concepts, dim_y, dim_c, ...) come from the loader, exactly as
    # in training; we only need one to build the architecture.
    _, model_kwargs, _ = get_funnybirds(
        train=False, batch_size=1, pkls_dir=cfg["data"]["pkls_dir"],
        img_size=cfg["data"].get("img_size", 224), return_nuisances=False,
    )
    model = get_model(**model_kwargs, **cfg["model"])
    res = MCBM / "results" / config_basename / str(seed)
    ckpt = (res / "models" / f"epoch_{epoch}.pt") if epoch else _latest_epoch_ckpt(res / "models")
    state = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"[grounding] loaded {ckpt}  (model_type={cfg['model']['model_type']})")
    return model, model_kwargs["n_concepts"]


@torch.inference_mode()
def concept_probs(model, imgs: torch.Tensor, n_concepts: int, device) -> np.ndarray:
    """imgs (B,3,H,W) -> (B, n_concepts) predicted concept probabilities."""
    c_dummy = torch.zeros(imgs.shape[0], n_concepts, device=device)  # unused by c_preds
    out = model(imgs.to(device), c_dummy)
    cp = out["c_preds"]                       # (B, n_concepts, dim_c=1)
    return cp[..., 0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config basename, e.g. funnybirds-cbm")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=None, help="default: latest saved")
    ap.add_argument("--funnybirds-root", required=True)
    ap.add_argument("--pkls", required=True, help="funnybirds_processed dir (for concept schema is via root)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N test images (0=all)")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    import funnybirds_concepts as fbc
    fb = Path(args.funnybirds_root)
    parts = fbc.load_parts(fb)                         # OrderedDict part -> variants
    lut = fbc.build_part_lookup(parts)
    spans = fbc.group_slices(parts)                    # part -> (a,b) in 26-vec
    part_keys = list(parts.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, n_concepts = load_model(args.config, args.seed, args.epoch, device)

    tfm = T.Compose([
        T.Resize(int(args.img_size * 256 / 224)),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])

    def load_img(path: Path):
        return tfm(Image.open(path).convert("RGB"))

    params = json.loads((fb / "dataset_test.json").read_text())
    if args.limit:
        params = params[: args.limit]

    rows = []
    n_missing = 0
    for idx, entry in enumerate(params):
        c = int(entry["class_idx"])
        gt = np.asarray(fbc.params_to_concept_vector(parts, lut, entry))   # 26-d one-hot
        stem = f"{idx:06d}"
        idir = fb / "test_interventions" / str(c) / stem
        # intact baseline = the ALL-PARTS-KEPT intervention image (same render
        # pipeline/background as the removed ones), so intact vs removed differ
        # ONLY by the part -- no background confound. Fall back to the plain test
        # image if the full-parts render is absent.
        allkept = idir / ("body_" + "_".join(sorted(part_keys)) + ".png")
        intact_path = allkept if allkept.exists() else (fb / "test" / str(c) / f"{stem}.png")
        if not intact_path.exists():
            n_missing += 1
            continue

        # build the batch: intact + one image per PRESENT part removed
        present = [p for p in part_keys if gt[spans[p][0]:spans[p][1]].sum() > 0]
        variants = [("__intact__", intact_path)]
        for p in present:
            keep = sorted(set(part_keys) - {p})
            variants.append((p, idir / ("body_" + "_".join(keep) + ".png")))
        imgs, tags = [], []
        for tag, path in variants:
            if not path.exists():
                n_missing += 1
                continue
            imgs.append(load_img(path)); tags.append(tag)
        if "__intact__" not in tags:
            continue
        probs = concept_probs(model, torch.stack(imgs), n_concepts, device)
        pintact = probs[tags.index("__intact__")]

        for p in present:
            if p not in tags:
                continue
            a, b = spans[p]
            typ = a + int(np.argmax(gt[a:b]))            # species-typical concept idx
            rows.append({
                "image_idx": idx, "class_idx": c, "part": p,
                "typ_concept": typ,
                "p_intact": float(pintact[typ]),
                "p_removed": float(probs[tags.index(p)][typ]),
            })

    df = pd.DataFrame(rows)
    df["drop"] = df["p_intact"] - df["p_removed"]
    if df.empty or not np.isfinite(df[["p_intact", "p_removed"]].to_numpy()).any():
        print(f"\n[grounding] *** WARNING: {args.config} s{args.seed} produced NON-FINITE "
              f"concept probs (NaN) -> this checkpoint DIVERGED in training. Retrain it. "
              f"Writing the parquet for provenance; collect_backwash will exclude it. ***")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    # ---- summary ----
    def agg(g):
        pi, pr = g["p_intact"].mean(), g["p_removed"].mean()
        grounding = (pi - pr) / pi if pi > 1e-6 else np.nan
        return pd.Series({"n": len(g), "p_intact": pi, "p_removed": pr,
                          "drop": pi - pr, "grounding": grounding,
                          "backwash": 1 - grounding})
    print(f"\n[grounding] {len(df)} (image,part) rows; {n_missing} missing images skipped")
    print("\n=== per-part (backwash = retained prob of a REMOVED part; 1=full backwash, 0=grounded) ===")
    per = df.groupby("part").apply(agg)
    print(per.round(3).to_string())
    o = agg(df)
    print(f"\n=== OVERALL  p_intact={o.p_intact:.3f}  p_removed={o.p_removed:.3f}  "
          f"grounding={o.grounding:.3f}  BACKWASH={o.backwash:.3f} ===")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
