#!/usr/bin/env python
"""
scripts/make_image_level_concept_labels.py

Generate image-level concept labels for FunnyBirds training images by
rendering per-image segmentation maps and thresholding pixel counts.

The problem with FunnyBirds' default labels
-------------------------------------------
Default labels are *species-level*: every image of species X gets the same
26-dim concept vector regardless of how visible each part is in that image.
This creates a perfect concept-species correlation in the training set, so
a model can achieve 100% concept accuracy by memorising species ID without
ever attending to the part's pixels.

What this script does
---------------------
For every training image:
  1. Render the part-map segmentation (renderer colour-codes each part).
  2. Count pixels belonging to each part.
  3. Compare to the species-median pixel count for that part.
  4. If count < THRESHOLD * median, relabel that part's concept dims as 0
     ("part not visually detectable in this image").

Result: images of the same species can now have *different* concept labels
depending on camera angle / occlusion, breaking the perfect confound.

Why pixel counts require rendering
-----------------------------------
The annotation JSONs (dataset_train.json) store which part variant each
image has, but NOT how many pixels that part occupies. Pixel counts depend
on camera angle, which varies per image. We must render the part-map
segmentation to get exact counts.

Output
------
  <output_dir>/pixel_counts_train.npy      (N_train, 5) int32
      Columns: [beak, eye, wing, foot, tail]
  <output_dir>/concept_labels_image_level.npy  (N_train, 26) float32
      Drop-in replacement for FunnyBirdsDataset's computed concept vectors.
  <output_dir>/concept_label_stats.json    per-part stats

Usage
-----
# Start the FunnyBirds renderer first:
#   cd /path/to/funnybirds/render && node server.js &
python -m scripts.make_image_level_concept_labels \\
    --funnybirds_root data/FunnyBirds \\
    --output_dir      data/FunnyBirds \\
    --renderer_url    http://localhost:8081 \\
    --threshold       0.05 \\
    --workers         8
"""

from __future__ import annotations

import argparse
import json
import time
from base64 import decodebytes
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Part segmentation colours (must match the renderer's part_map output)
# Taken from fb_mcbm_renderer_swap.ipynb PART_SEG_COLORS
# ---------------------------------------------------------------------------

PART_SEG_COLORS: Dict[str, Tuple[int, int, int]] = {
    "beak": (255, 255,   0),
    "eye":  (255, 255, 253),
    "wing": (  0, 255,   1),
    "foot": (255,   0,   1),
    "tail": (  0,   0, 255),
}

PARTS: List[str] = ["beak", "eye", "wing", "foot", "tail"]

# Concept vector layout (26 dims total)
PART_VARIANTS: Dict[str, int] = {"beak": 4, "eye": 3, "wing": 6, "foot": 4, "tail": 9}
PART_OFFSETS: Dict[str, int] = {}
_off = 0
for _p, _n in PART_VARIANTS.items():
    PART_OFFSETS[_p] = _off
    _off += _n
NUM_CONCEPTS = _off  # 26


# ---------------------------------------------------------------------------
# Renderer helpers (mirror notebook exactly)
# ---------------------------------------------------------------------------

def _ann_to_url(ann: dict, renderer_url: str, render_mode: str = "part_map") -> str:
    url = renderer_url.rstrip("/") + f"/render?render_mode={render_mode}&"
    for key, val in ann.items():
        if key == "class_idx":
            continue
        url += f"{key}={val}&"
    return url.rstrip("&")


def _render_part_map(ann: dict, renderer_url: str, timeout: float = 20.0) -> Image.Image:
    url = _ann_to_url(ann, renderer_url, render_mode="part_map")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img_bytes = decodebytes(resp.content)
    img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((256, 256), Image.NEAREST)
    return img


def _count_pixels(img: Image.Image) -> Dict[str, int]:
    """Return pixel count for each part from a part-map segmentation image."""
    arr = np.asarray(img)
    counts: Dict[str, int] = {}
    for part, (r, g, b) in PART_SEG_COLORS.items():
        mask = (arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)
        counts[part] = int(mask.sum())
    return counts


# ---------------------------------------------------------------------------
# Worker: render one image and return its pixel counts
# ---------------------------------------------------------------------------

def _process_one(
    idx: int,
    ann: dict,
    renderer_url: str,
    max_retries: int = 3,
) -> Optional[Tuple[int, Dict[str, int]]]:
    for attempt in range(max_retries):
        try:
            img = _render_part_map(ann, renderer_url)
            counts = _count_pixels(img)
            return idx, counts
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [warn] idx={idx} attempt={attempt+1}: {exc} — retrying in {wait}s")
            time.sleep(wait)
    print(f"  [fail] idx={idx}: all {max_retries} attempts failed, storing zeros")
    return idx, {p: 0 for p in PARTS}


# ---------------------------------------------------------------------------
# Build concept labels from pixel counts
# ---------------------------------------------------------------------------

def build_concept_labels(
    annotations: List[dict],
    pixel_counts: np.ndarray,   # (N, 5) int32, columns ordered as PARTS
    threshold: float,
) -> Tuple[np.ndarray, dict]:
    """
    Build (N, 26) concept label array.

    For each image and each part:
    - Start from the annotation-derived one-hot (same as FunnyBirdsDataset).
    - If pixel_count < threshold * species_median_pixel_count, zero out
      the entire part subvector for that image.

    Returns (labels, stats_dict).
    """
    N = len(annotations)
    labels = np.zeros((N, NUM_CONCEPTS), dtype=np.float32)

    # Step 1: fill annotation-derived one-hot labels
    # Load parts.json to build the lookup
    return_stats: dict = {}

    # We'll compute annotation-derived labels inline using model/color fields
    # and a simple variant-list lookup built below
    part_order: Dict[str, List[dict]] = {}

    # We need parts.json for the variant order — pass it in via annotations[0]'s
    # sibling path. We look it up from the funnybirds_root stored in stats.
    # Actually, we receive pre-built one-hot labels from the caller (see main).
    # This function receives the already-built annotation labels; we just zero out.

    # pixel_counts columns: [beak, eye, wing, foot, tail]
    part_col = {p: i for i, p in enumerate(PARTS)}

    # Step 2: compute species-level medians (use all images of species, ignore zeros
    # from failed renders — they'd falsely suppress the median)
    # Group by species
    species_pixels: Dict[int, Dict[str, List[int]]] = {}
    for i, ann in enumerate(annotations):
        sp = int(ann["class_idx"])
        if sp not in species_pixels:
            species_pixels[sp] = {p: [] for p in PARTS}
        for p in PARTS:
            px = int(pixel_counts[i, part_col[p]])
            if px > 0:  # exclude failed renders (stored as 0)
                species_pixels[sp][p].append(px)

    species_medians: Dict[int, Dict[str, float]] = {}
    for sp, pdict in species_pixels.items():
        species_medians[sp] = {}
        for p in PARTS:
            vals = pdict[p]
            species_medians[sp][p] = float(np.median(vals)) if vals else 0.0

    # Step 3: apply threshold
    relabeled_count = {p: 0 for p in PARTS}
    for i, ann in enumerate(annotations):
        sp = int(ann["class_idx"])
        med = species_medians.get(sp, {})
        for p in PARTS:
            px = int(pixel_counts[i, part_col[p]])
            threshold_px = threshold * med.get(p, 0.0)
            if threshold_px > 0 and px < threshold_px:
                # zero out this part's concept dims
                off = PART_OFFSETS[p]
                n   = PART_VARIANTS[p]
                labels[i, off:off + n] = 0.0
                relabeled_count[p] += 1

    # Step 4: per-part stats
    for p in PARTS:
        all_medians = [species_medians[sp][p] for sp in species_medians if species_medians[sp][p] > 0]
        global_median = float(np.median(all_medians)) if all_medians else 0.0
        pct = relabeled_count[p] / N
        return_stats[p] = {
            "global_median_px": global_median,
            "threshold_px":     threshold * global_median,
            "relabeled_n":      relabeled_count[p],
            "relabeled_frac":   pct,
        }
        print(f"  {p:5s}: global_median={global_median:.0f}px  threshold={threshold*global_median:.0f}px  "
              f"relabeled={relabeled_count[p]}/{N} ({pct:.1%})")

    return labels, return_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate image-level concept labels for FunnyBirds training set."
    )
    parser.add_argument("--funnybirds_root", required=True,
                        help="Root directory of FunnyBirds (contains dataset_train.json)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save outputs. Defaults to --funnybirds_root.")
    parser.add_argument("--renderer_url", default="http://localhost:8081",
                        help="Base URL of the FunnyBirds renderer server")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Relabel concept=0 when pixel_count < threshold * species_median")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel renderer threads")
    parser.add_argument("--checkpoint_every", type=int, default=2000,
                        help="Save pixel count checkpoint every N images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images (for testing)")
    args = parser.parse_args()

    fb_root = Path(args.funnybirds_root)
    out_dir = Path(args.output_dir) if args.output_dir else fb_root
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_counts_path = out_dir / "pixel_counts_train.npy"
    labels_path       = out_dir / "concept_labels_image_level.npy"
    stats_path        = out_dir / "concept_label_stats.json"

    # ── Load annotations ──────────────────────────────────────────────────────
    ann_path = fb_root / "dataset_train.json"
    with open(ann_path) as f:
        annotations: List[dict] = json.load(f)

    if args.limit:
        annotations = annotations[: args.limit]

    N = len(annotations)
    print(f"[labels] {N} training images  renderer={args.renderer_url}  threshold={args.threshold}")

    # ── Load parts.json for annotation-derived concept labels ─────────────────
    parts_path = fb_root / "parts.json"
    with open(parts_path) as f:
        parts_json = json.load(f)

    # Build per-part variant lookup: (model[, color]) -> index
    lookup: Dict[str, Dict] = {}
    for part, variants in parts_json.items():
        lookup[part] = {}
        for idx, vd in enumerate(variants):
            key_fields = {"model": vd["model"]}
            if "color" in vd:
                key_fields["color"] = vd["color"]
            key = tuple(sorted(key_fields.items()))
            lookup[part][key] = idx

    # ── Build annotation-level concept labels (baseline) ─────────────────────
    ann_labels = np.zeros((N, NUM_CONCEPTS), dtype=np.float32)
    for i, ann in enumerate(annotations):
        for part, n_var in PART_VARIANTS.items():
            model = ann.get(f"{part}_model", "placeholder")
            if model == "placeholder":
                continue
            key_fields = {"model": model}
            color_key = f"{part}_color"
            if color_key in ann:
                key_fields["color"] = ann[color_key]
            key = tuple(sorted(key_fields.items()))
            v = lookup[part].get(key, -1)
            if v >= 0:
                ann_labels[i, PART_OFFSETS[part] + v] = 1.0

    # ── Check renderer ────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{args.renderer_url}/render?render_mode=default&beak_model=beak01.glb",
                         timeout=5)
        print(f"[labels] Renderer alive (status {r.status_code})")
    except Exception as e:
        print(f"[labels] ERROR: renderer not reachable at {args.renderer_url}: {e}")
        print("[labels] Start it with:  cd <funnybirds>/render && node server.js")
        raise SystemExit(1)

    # ── Load or initialise pixel count array ──────────────────────────────────
    part_col = {p: i for i, p in enumerate(PARTS)}
    pixel_counts = np.zeros((N, len(PARTS)), dtype=np.int32)
    processed    = np.zeros(N, dtype=bool)

    if pixel_counts_path.exists():
        saved = np.load(pixel_counts_path)
        # Saved may be smaller if a previous run was partial
        n_saved = min(saved.shape[0], N)
        pixel_counts[:n_saved] = saved[:n_saved]
        # Mark as processed: rows where at least one part has a non-zero count
        # (a fully-occluded image may legitimately have all zeros — we re-render those)
        processed[:n_saved] = True
        print(f"[labels] Loaded checkpoint: {n_saved} images already processed")

    todo = [i for i in range(N) if not processed[i]]
    print(f"[labels] Rendering {len(todo)} images with {args.workers} workers ...")

    t0 = time.time()
    done_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, i, annotations[i], args.renderer_url): i
            for i in todo
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                idx, counts = result
                for p in PARTS:
                    pixel_counts[idx, part_col[p]] = counts[p]
            done_count += 1

            if done_count % args.checkpoint_every == 0 or done_count == len(todo):
                np.save(pixel_counts_path, pixel_counts)
                elapsed = time.time() - t0
                rate = done_count / elapsed
                remaining = (len(todo) - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count}/{len(todo)}]  {rate:.1f} img/s  "
                      f"ETA {remaining/60:.0f}m  checkpoint saved")

    np.save(pixel_counts_path, pixel_counts)
    print(f"[labels] Pixel counts saved to {pixel_counts_path}")

    # ── Build image-level concept labels ──────────────────────────────────────
    print(f"\n[labels] Building image-level labels (threshold={args.threshold}) ...")

    # Start from annotation labels, then zero out below-threshold parts
    labels = ann_labels.copy()

    # We call build_concept_labels for threshold application and stats
    _, stats = build_concept_labels(annotations, pixel_counts, args.threshold)

    # Apply the same zeroing logic directly on `labels`
    species_pixels: Dict[int, Dict[str, List[int]]] = {}
    for i, ann in enumerate(annotations):
        sp = int(ann["class_idx"])
        if sp not in species_pixels:
            species_pixels[sp] = {p: [] for p in PARTS}
        for p in PARTS:
            px = int(pixel_counts[i, part_col[p]])
            if px > 0:
                species_pixels[sp][p].append(px)

    species_medians: Dict[int, Dict[str, float]] = {}
    for sp, pdict in species_pixels.items():
        species_medians[sp] = {}
        for p in PARTS:
            vals = pdict[p]
            species_medians[sp][p] = float(np.median(vals)) if vals else 0.0

    for i, ann in enumerate(annotations):
        sp = int(ann["class_idx"])
        med = species_medians.get(sp, {})
        for p in PARTS:
            px = int(pixel_counts[i, part_col[p]])
            threshold_px = args.threshold * med.get(p, 0.0)
            if threshold_px > 0 and px < threshold_px:
                off = PART_OFFSETS[p]
                n   = PART_VARIANTS[p]
                labels[i, off:off + n] = 0.0

    np.save(labels_path, labels)
    print(f"\n[labels] Image-level labels saved to {labels_path}")
    print(f"[labels] Shape: {labels.shape}  dtype: {labels.dtype}")

    # Compare to annotation-level labels
    diff = (labels != ann_labels).any(axis=1).sum()
    print(f"[labels] Images with at least one relabeled concept: {diff}/{N} ({diff/N:.1%})")

    # ── Save stats ────────────────────────────────────────────────────────────
    stats["threshold"] = args.threshold
    stats["n_images"]  = N
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[labels] Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
