#!/usr/bin/env python3
"""Validity check for the deletion test — NO model, NO GPU, seconds.

Answers "was the part actually removed, and how often does removal change nothing?"
by diffing the pre-rendered FunnyBirds intervention images directly:

  intact  = test_interventions/<c>/<i>/body_<all parts>.png
  removed = test_interventions/<c>/<i>/body_<all but p>.png
  changed_frac = fraction of pixels that differ (|intact-removed| > eps)

Per part it reports:
  mean_changed_frac  how much the render changes on average when p is removed
  frac_noop          share of (image,part) where removal changed ~nothing
                     (part invisible/occluded in intact -> retained_frac=1 is TRIVIAL)

If a part has a high frac_noop, its deletion-backwash is inflated by no-op removals
and must be recomputed on visible-only rows (grounding_deletion writes changed_frac
per row for exactly this gate).

  python analysis/verify_deletion.py --funnybirds-root $CURATED_DATA/FunnyBirds
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data" / "funnybirds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnybirds-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N test images (0=all)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--eps", type=float, default=3.0, help="per-pixel 0-255 diff threshold")
    args = ap.parse_args()

    import funnybirds_concepts as fbc
    fb = Path(args.funnybirds_root)
    parts = fbc.load_parts(fb)
    lut = fbc.build_part_lookup(parts)
    spans = fbc.group_slices(parts)
    part_keys = list(parts.keys())

    params = json.loads((fb / "dataset_test.json").read_text())
    if args.limit:
        params = params[: args.limit]

    def load(path):
        im = Image.open(path).convert("RGB").resize((args.size, args.size))
        return np.asarray(im, dtype=np.int16)

    rows = {p: [] for p in part_keys}
    n_missing = 0
    for idx, entry in enumerate(params):
        c = int(entry["class_idx"])
        gt = np.asarray(fbc.params_to_concept_vector(parts, lut, entry))
        idir = fb / "test_interventions" / str(c) / f"{idx:06d}"
        allkept = idir / ("body_" + "_".join(sorted(part_keys)) + ".png")
        if not allkept.exists():
            n_missing += 1
            continue
        intact = load(allkept)
        present = [p for p in part_keys if gt[spans[p][0]:spans[p][1]].sum() > 0]
        for p in present:
            keep = sorted(set(part_keys) - {p})
            rp = idir / ("body_" + "_".join(keep) + ".png")
            if not rp.exists():
                continue
            removed = load(rp)
            changed = float((np.abs(intact - removed).max(-1) > args.eps).mean())
            rows[p].append(changed)

    print(f"[verify] {sum(len(v) for v in rows.values())} (image,present-part) pairs; "
          f"{n_missing} images missing all-kept render")
    print(f"\n{'part':6s} {'n':>6s} {'mean_changed':>12s} {'frac_noop(<0.1%)':>16s}")
    for p in part_keys:
        v = np.array(rows[p])
        if not len(v):
            print(f"{p:6s} {'0':>6s}"); continue
        noop = float((v <= 1e-3).mean())
        print(f"{p:6s} {len(v):>6d} {v.mean():>12.4f} {noop:>16.3f}")
    print("\nRead: high frac_noop for a part => many 'removals' change nothing (part was "
          "occluded) => that part's retained_frac is inflated; use the visible-only gate.")


if __name__ == "__main__":
    main()
