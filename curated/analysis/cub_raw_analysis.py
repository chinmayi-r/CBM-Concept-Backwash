#!/usr/bin/env python3
"""CUB RAW-annotation analysis (no processed pkls, no model, CPU).

Reads the raw CUB-200-2011 image-level attribute labels directly and answers the
CUB story's two load-bearing questions:

  1. Do CUB attributes VARY within a species at the image level? (species-constancy)
     -> if yes, the matched-pair recall gap is powered on CUB (FunnyBirds was 0%).
  2. What does the CBM "majority-vote to class level" standardization DO to that
     variation? We majority-vote each attribute per species (Koh et al.'s recipe)
     and show it collapses the within-species signal to 0 -- the §3b critique,
     measured. Plus the count of genuinely AMBIGUOUS (species,attr) pairs (present
     in 30-70% of a species' images) -- the "belly looks white or yellow depending
     on lighting" case.

Runs on all 200 classes and on the first-70 (CUB70) subset.

  python analysis/cub_raw_analysis.py --cub-root <.../data/CUB_200_2011> \
      --out $CURATED_DATA/cub_data_analysis
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np


def _guess_root(cli):
    if cli:
        return Path(cli)
    cd = os.environ.get("CURATED_DATA", "")
    for c in [Path(cd).parent / "data" / "CUB_200_2011",
              Path(cd) / "CUB_200_2011",
              Path.home() / "cv_emergence_project" / "data" / "CUB_200_2011"]:
        if c.exists():
            return c
    return Path(cli or "CUB_200_2011")


def _load_image_attr(path: Path):
    """image_attribute_labels.txt -> (img_ids, attr_ids, present) robust to the
    known CUB rows with extra whitespace tokens (take the first 3 ints)."""
    imgs, attrs, pres = [], [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                imgs.append(int(p[0])); attrs.append(int(p[1])); pres.append(int(p[2]))
    return np.array(imgs), np.array(attrs), np.array(pres, dtype=np.int8)


def analyse(name, M, y, lines):
    classes = np.unique(y)
    lines.append(f"\n{'='*66}\n### {name}: {M.shape[0]} images · {len(classes)} species · {M.shape[1]} attributes\n{'='*66}")

    # within-species std on RAW image-level labels
    within = np.array([M[y == c].std(0) for c in classes])          # (n_species, n_attr)
    frac_const_raw = float(np.mean(within == 0))
    lines.append(f"RAW image-level: (species,attr) with within-species std==0: {frac_const_raw:.4f} "
                 f"-> {(1-frac_const_raw)*100:.1f}% VARY within a species")

    # prevalence-per-species, to flag genuinely ambiguous pairs
    persp = np.array([M[y == c].mean(0) for c in classes])          # P(present) per (species,attr)
    ambiguous = float(np.mean((persp >= 0.3) & (persp <= 0.7)))
    lines.append(f"AMBIGUOUS (species,attr) pairs (present in 30-70% of a species' imgs): "
                 f"{ambiguous:.4f}  ('belly white or yellow?' cases)")

    # majority-vote to class level (Koh et al. standardization) -> constant by construction
    mv = (persp >= 0.5).astype(np.int8)
    # after MV every image in a species shares mv -> within-species std == 0 for ALL pairs
    lines.append("MAJORITY-VOTE (CBM class-level labels): within-species std==0 for 100% of pairs "
                 "by construction -> the standardization ERASES the raw variation above.")
    # how many (species,attr) labels get FLIPPED for the minority images
    flips = 0; total = 0
    for i, c in enumerate(classes):
        block = M[y == c]                                            # imgs x attr
        flips += int((block != mv[i]).sum()); total += block.size
    lines.append(f"labels changed by majority-voting: {flips}/{total} image-attribute cells "
                 f"({100*flips/total:.1f}%) -- these images are RELABELED to the species-typical value.")

    verdict = ("VERDICT: raw CUB has real within-species variation -> recall gap is powered here, "
               "AND majority-voting removes it -> the §3b label-standardization critique is exactly this."
               if frac_const_raw < 0.999 else
               "VERDICT: even raw CUB is species-constant here (unexpected -> check attribute subset).")
    lines.append(verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-root", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = _guess_root(args.cub_root)
    attr_file = root / "attributes" / "image_attribute_labels.txt"
    if not attr_file.exists():
        alt = root / "image_attribute_labels.txt"
        attr_file = alt if alt.exists() else attr_file
    lbl_file = root / "image_class_labels.txt"
    if not attr_file.exists() or not lbl_file.exists():
        raise SystemExit(f"missing {attr_file} or {lbl_file}\n(pass --cub-root <.../CUB_200_2011>)")

    print(f"[cub_raw] reading {attr_file} ...")
    imgs, attrs, pres = _load_image_attr(attr_file)
    nimg, natt = int(imgs.max()), int(attrs.max())
    M = np.zeros((nimg, natt), dtype=np.int8)
    M[imgs - 1, attrs - 1] = pres

    cls = np.loadtxt(lbl_file, dtype=np.int64)          # img_id, class_id
    y = np.zeros(nimg, dtype=np.int64)
    y[cls[:, 0] - 1] = cls[:, 1]

    lines = []
    analyse("full CUB (200)", M, y, lines)
    # CUB70 = first 70 classes
    keep = y <= 70
    analyse("CUB70 (first 70 classes)", M[keep], y[keep], lines)

    text = "\n".join(lines); print(text)
    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "CUB_RAW_SUMMARY.txt").write_text(text)
        print(f"\n[cub_raw] wrote {out/'CUB_RAW_SUMMARY.txt'}")


if __name__ == "__main__":
    main()
