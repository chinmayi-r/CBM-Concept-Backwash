#!/usr/bin/env python3
"""Small, explicitly limited CUB70 beak/tail insertion pilot.

This is not a renderer-quality intervention.  It asks a narrower question:
when clearly visible pixels from a different value of the *same* CUB attribute
family are pasted into a visible beak or tail, does the donor-minus-source raw
concept margin move toward the donor value?

Each target image is scored in four forms:

  original, target deleted, source-value paste control, donor-value paste.

The source-value control uses the identical crop/resize/paste pipeline, so the
primary response is:

  (donor z - source z) after donor paste
  - (donor z - source z) after source-value control paste.

A positive response shows sensitivity to the donor pixels.  A positive response
with a still-negative final margin is only a *candidate* analogue of FunnyBird
backwash: the new pixels helped, but did not beat the old source/context.  Pose,
imperfect masks, resizing, and donor-image appearance remain alternatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
for path in (HERE, CURATED / "data" / "cub70"):
    sys.path.insert(0, str(path))

from cub70_parts import attribute_type
from paired_mask_deletion import (
    cub_masks,
    cub_records,
    delete_region,
    model_and_dataset,
    overlay,
    score,
    sha256_image,
)


PARTS = ("beak", "tail")


def mask_geometry(mask: np.ndarray) -> dict[str, float | tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty mask")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width, height = x1 - x0 + 1, y1 - y0 + 1
    return {
        "bbox": (x0, y0, x1 + 1, y1 + 1),
        "area": float(mask.sum()),
        "aspect": width / max(height, 1),
        "fill": float(mask[y0:y1 + 1, x0:x1 + 1].mean()),
    }


def geometry_distance(a: dict, b: dict) -> float:
    return (abs(math.log(max(a["aspect"], 1e-6) / max(b["aspect"], 1e-6)))
            + abs(math.log(max(a["fill"], 1e-6) / max(b["fill"], 1e-6))))


def nearest_fill(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill crop pixels outside the donor mask with the nearest donor-part pixel."""
    if valid.all():
        return rgb
    try:
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(~valid, return_indices=True)
        return rgb[indices[0], indices[1]]
    except Exception:
        # Conservative fallback: use the median donor-part colour.  The audit
        # records this implementation globally; visual sheets remain mandatory.
        out = rgb.copy()
        out[~valid] = np.median(rgb[valid], axis=0).astype(np.uint8)
        return out


def paste_part(target_image: Image.Image, target_mask: np.ndarray,
               donor_image: Image.Image, donor_mask: np.ndarray) -> Image.Image:
    """Resize donor part texture into the target mask with a soft one-pixel edge."""
    tg = mask_geometry(target_mask)
    dg = mask_geometry(donor_mask)
    tx0, ty0, tx1, ty1 = tg["bbox"]
    dx0, dy0, dx1, dy1 = dg["bbox"]

    donor_rgb = np.asarray(donor_image.convert("RGB"))[dy0:dy1, dx0:dx1]
    donor_valid = donor_mask[dy0:dy1, dx0:dx1]
    donor_rgb = nearest_fill(donor_rgb, donor_valid)
    width, height = tx1 - tx0, ty1 - ty0
    texture = Image.fromarray(donor_rgb).resize((width, height), Image.Resampling.BILINEAR)

    canvas = target_image.convert("RGB").copy()
    texture_canvas = canvas.copy()
    texture_canvas.paste(texture, (tx0, ty0))
    alpha = Image.fromarray((target_mask.astype(np.uint8) * 255), "L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.0))
    return Image.composite(texture_canvas, canvas, alpha)


def sheet(path: Path, panels: list[Image.Image], labels: list[str]) -> None:
    width, height = panels[0].size
    out = Image.new("RGB", (width * len(panels), height + 30), "white")
    draw = ImageDraw.Draw(out)
    for col, (label, panel) in enumerate(zip(labels, panels)):
        draw.text((col * width + 4, 6), label, fill="black")
        out.paste(panel.convert("RGB"), (col * width, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cub70-cbm")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=250)
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs-per-part", type=int, default=40)
    ap.add_argument("--examples-per-part", type=int, default=8)
    ap.add_argument("--min-beak-frac", type=float, default=.0007)
    ap.add_argument("--min-tail-frac", type=float, default=.004)
    ap.add_argument("--max-geometry-distance", type=float, default=.85)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run inside an already allocated GPU session")
    if not args.data_root:
        raise ValueError("set CURATED_DATA or pass --data-root")
    data_root = Path(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    example_dir = out.parent / f"{out.stem}_examples"

    device = torch.device("cuda")
    model, n_concepts, dataset, cfg = model_and_dataset(
        args.config, args.seed, args.epoch, device)
    records = cub_records(dataset, cfg, args.seed, data_root)
    archive = data_root / "cub70" / "masks" / "AnnotationMasksPerclass"

    candidates: list[dict] = []
    counts = Counter()
    for index, class_label, image_path, _, positives in tqdm(records, desc="index clear masks"):
        if not image_path.exists():
            counts["missing_rgb"] += 1
            continue
        image = Image.open(image_path).convert("RGB")
        by_part = defaultdict(list)
        for concept_idx, concept_name, part in positives:
            if part in PARTS:
                by_part[part].append((concept_idx, concept_name))
        for part, concepts in by_part.items():
            target, _ = cub_masks(archive, class_label, image_path.stem, part, image.size)
            fraction = float(target.mean())
            threshold = args.min_beak_frac if part == "beak" else args.min_tail_frac
            if fraction < threshold:
                counts[f"{part}_mask_too_small"] += 1
                continue
            geometry = mask_geometry(target)
            for concept_idx, concept_name in concepts:
                candidates.append({
                    "index": index, "class_label": class_label, "image_path": image_path,
                    "part": part, "concept_idx": concept_idx,
                    "concept_name": concept_name,
                    "attribute_type": attribute_type(concept_name),
                    "mask": target, "fraction": fraction, "geometry": geometry,
                })
                counts[f"{part}_positive_clear_candidates"] += 1

    by_type_value = defaultdict(list)
    for candidate in candidates:
        by_type_value[(candidate["part"], candidate["attribute_type"],
                       candidate["concept_idx"])].append(candidate)

    selected = []
    per_part = Counter()
    used_targets = set()
    rng = np.random.default_rng(args.seed)
    ordered = list(range(len(candidates)))
    rng.shuffle(ordered)
    for ci in ordered:
        target = candidates[ci]
        part = target["part"]
        if per_part[part] >= args.max_pairs_per_part or target["index"] in used_targets:
            continue
        same_family = [c for c in candidates
                       if c["part"] == part
                       and c["attribute_type"] == target["attribute_type"]
                       and c["concept_idx"] != target["concept_idx"]
                       and c["index"] != target["index"]]
        source_controls = [c for c in by_type_value[(part, target["attribute_type"],
                                                     target["concept_idx"])]
                           if c["index"] != target["index"]]
        if not same_family or not source_controls:
            counts["no_donor_or_source_control"] += 1
            continue

        def rank(c):
            same_species_penalty = 0 if c["class_label"] == target["class_label"] else 1
            return (same_species_penalty, geometry_distance(target["geometry"], c["geometry"]),
                    c["index"])

        donor = min(same_family, key=rank)
        source_control = min(source_controls, key=rank)
        donor_distance = geometry_distance(target["geometry"], donor["geometry"])
        source_distance = geometry_distance(target["geometry"], source_control["geometry"])
        if max(donor_distance, source_distance) > args.max_geometry_distance:
            counts["geometry_rejected"] += 1
            continue
        selected.append((target, source_control, donor, donor_distance, source_distance))
        used_targets.add(target["index"])
        per_part[part] += 1

    rows = []
    example_counts = Counter()
    for target, source_control, donor, donor_distance, source_distance in tqdm(
            selected, desc="score insertion pilot"):
        target_image = Image.open(target["image_path"]).convert("RGB")
        source_image = Image.open(source_control["image_path"]).convert("RGB")
        donor_image = Image.open(donor["image_path"]).convert("RGB")
        deleted, deletion_method = delete_region(target_image, target["mask"])
        source_paste = paste_part(target_image, target["mask"], source_image,
                                  source_control["mask"])
        donor_paste = paste_part(target_image, target["mask"], donor_image, donor["mask"])
        variants = [target_image, deleted, source_paste, donor_paste]
        values = score(model, dataset.transform, variants, n_concepts, device)
        src = target["concept_idx"]
        don = donor["concept_idx"]
        margins = values[:, don] - values[:, src]
        response_from_original = float(margins[3] - margins[0])
        response_vs_source_control = float(margins[3] - margins[2])
        rows.append({
            "config": args.config, "seed": args.seed, "epoch": args.epoch,
            "image_index": target["index"], "image": target["image_path"].stem,
            "class_label": target["class_label"], "part": target["part"],
            "attribute_type": target["attribute_type"],
            "source_concept_idx": src, "source_concept": target["concept_name"],
            "donor_concept_idx": don, "donor_concept": donor["concept_name"],
            "source_control_image": source_control["image_path"].stem,
            "source_control_class": source_control["class_label"],
            "donor_image": donor["image_path"].stem,
            "donor_class": donor["class_label"],
            "donor_same_species": donor["class_label"] == target["class_label"],
            "source_control_same_species": source_control["class_label"] == target["class_label"],
            "mask_fraction": target["fraction"],
            "donor_geometry_distance": donor_distance,
            "source_geometry_distance": source_distance,
            "deletion_method": deletion_method,
            "z_source_original": float(values[0, src]),
            "z_donor_original": float(values[0, don]),
            "z_source_deleted": float(values[1, src]),
            "z_donor_deleted": float(values[1, don]),
            "z_source_source_control": float(values[2, src]),
            "z_donor_source_control": float(values[2, don]),
            "z_source_donor_insert": float(values[3, src]),
            "z_donor_donor_insert": float(values[3, don]),
            "margin_original": float(margins[0]),
            "margin_deleted": float(margins[1]),
            "margin_source_control": float(margins[2]),
            "margin_donor_insert": float(margins[3]),
            "response_from_original": response_from_original,
            "response_vs_source_control": response_vs_source_control,
            "donor_wins_after_insert": bool(margins[3] > 0),
            "candidate_retained_source": bool(response_vs_source_control > 0 and margins[3] < 0),
            "original_sha256": sha256_image(target_image),
            "deleted_sha256": sha256_image(deleted),
            "source_control_sha256": sha256_image(source_paste),
            "donor_insert_sha256": sha256_image(donor_paste),
        })
        if example_counts[target["part"]] < args.examples_per_part:
            panels = [target_image, overlay(target_image, target["mask"]), deleted,
                      source_image, source_paste, donor_image, donor_paste]
            labels = ["target original", "target mask", "target deleted",
                      "source-value donor", "source-value paste",
                      "different-value donor", "different-value paste"]
            sheet(example_dir / f"{target['part']}_{target['index']:06d}.png",
                  panels, labels)
            example_counts[target["part"]] += 1

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no geometry-matched beak/tail pairs were found")
    required = set(PARTS)
    present = set(frame.part.unique())
    if present != required:
        raise RuntimeError(f"pilot coverage incomplete: expected {required}, got {present}")
    if (frame.groupby("part").size() < 10).any():
        raise RuntimeError("fewer than 10 usable pairs for at least one part")
    for column in ["deleted_sha256", "source_control_sha256", "donor_insert_sha256"]:
        if (frame[column] == frame.original_sha256).any():
            raise RuntimeError(f"at least one {column} image did not change")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric).all().all():
        raise RuntimeError("non-finite pilot output")

    frame.to_parquet(out, index=False)
    summary = frame.groupby("part").agg(
        n=("image", "size"),
        median_response_vs_source_control=("response_vs_source_control", "median"),
        positive_response_rate=("response_vs_source_control", lambda x: (x > 0).mean()),
        median_final_margin=("margin_donor_insert", "median"),
        donor_win_rate=("donor_wins_after_insert", "mean"),
        retained_source_candidate_rate=("candidate_retained_source", "mean"),
        same_species_donor_rate=("donor_same_species", "mean"),
    ).reset_index()
    summary.to_csv(out.with_suffix(".summary.csv"), index=False)
    audit = {
        "status": "COMPUTATION_PASS_VISUAL_REVIEW_REQUIRED",
        "claim_limit": "crude 2-D insertion sensitivity only; not renderer-quality causal swap",
        "config": args.config, "seed": args.seed, "epoch": args.epoch,
        "rows": len(frame), "parts": sorted(present),
        "selection_counts": dict(counts), "selected_by_part": dict(per_part),
        "examples": str(example_dir),
        "primary_metric": "margin_donor_insert - margin_source_control",
        "acceptance": [
            "inspect every saved sheet before interpreting scores",
            "donor response must be positive in direction and not be driven by one pair",
            "a negative final margin after positive response is candidate residual source/context only",
        ],
    }
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(f"[CUB70 BEAK/TAIL PILOT COMPUTATION PASS; VISUAL REVIEW REQUIRED] -> {out}")


if __name__ == "__main__":
    main()
