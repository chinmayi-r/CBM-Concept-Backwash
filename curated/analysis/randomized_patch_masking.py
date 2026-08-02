#!/usr/bin/env python3
"""Small, smooth randomized-mask dose response for FunnyBirds or CUB70.

This is deliberately weaker than a renderer swap.  It asks whether repeatedly
covering small regions inside a named part changes that part's exact concept
score more than the same amount of masking elsewhere.  CUB70 is never run by
the driver until this method passes its FunnyBird calibration.

For each positive exact concept on a visibly masked part we score:

* the original image;
* smooth patches centred inside the target part;
* an exact translated copy of that smooth mask on other bird pixels;
* another exact translated copy on background pixels.

Every mask is rendered with two fills (local blur and local mean colour), at
several target-coverage doses and repeated random placements.  Raw logits and
probabilities are both retained.  The script trains nothing and submits no job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch
import torchvision.transforms as T
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
MCBM = CURATED / "external" / "minimal_cbm"
for path in (MCBM, CURATED / "compat", CURATED / "data" / "funnybirds",
             CURATED / "data" / "cub70", HERE):
    sys.path.insert(0, str(path))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

from paired_mask_deletion import (
    cub_masks, cub_records, funnybird_mask, funnybird_records,
    model_and_dataset, score, sha256_image,
)


FILLS = ("local_blur", "local_mean")
LOCATIONS = ("target", "other_bird", "background")


def stable_seed(seed: int, image_index: int, part: str, repeat: int,
                location: str) -> int:
    payload = f"{seed}|{image_index}|{part}|{repeat}|{location}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    # MaxFilter requires an odd kernel.
    return np.asarray(image.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0


def add_gaussian(alpha: np.ndarray, x: int, y: int, sigma: float) -> None:
    radius = max(2, int(np.ceil(3 * sigma)))
    h, w = alpha.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], patch)


def shifted(alpha: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate without wraparound."""
    h, w = alpha.shape
    result = np.zeros_like(alpha)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    if sx0 >= sx1 or sy0 >= sy1:
        return result
    result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = alpha[sy0:sy1, sx0:sx1]
    return result


def translate_exact_mask(alpha: np.ndarray, support: np.ndarray,
                         rng: np.random.Generator,
                         min_supported_mass: float = .75) -> np.ndarray | None:
    """Place an exact translated copy mostly inside the requested support."""
    ys, xs = np.where(alpha > .02)
    sy, sx = np.where(support)
    if not len(xs) or not len(sx):
        return None
    centre_x, centre_y = int(np.round(np.average(xs, weights=alpha[ys, xs]))), \
        int(np.round(np.average(ys, weights=alpha[ys, xs])))
    order = rng.permutation(len(sx))[:min(len(sx), 800)]
    total = float(alpha.sum())
    best, best_fraction = None, -1.0
    for pick in order:
        candidate = shifted(alpha, int(sx[pick]) - centre_x,
                            int(sy[pick]) - centre_y)
        # A crop at the image edge changes the total damage and is rejected.
        if abs(float(candidate.sum()) - total) > max(1e-5 * total, 1e-5):
            continue
        fraction = float((candidate * support).sum()) / max(total, 1e-8)
        if fraction > best_fraction:
            best, best_fraction = candidate, fraction
        if fraction >= .98:
            break
    return best if best is not None and best_fraction >= min_supported_mass else None


def make_patch_masks(target: np.ndarray, bird: np.ndarray, doses: list[float],
                     sigma: float, rng: np.random.Generator) -> dict[str, list[np.ndarray]]:
    """Return nested masks: location -> one cumulative mask per dose.

    The target mask determines how many Gaussian patches are needed at each
    dose. Controls are translated copies with identical alpha mass and shape.
    """
    h, w = target.shape
    protected = dilate(target, max(2, int(np.ceil(3 * sigma))))
    supports = {
        "target": target,
        "other_bird": bird & ~protected,
        "background": ~dilate(bird, max(1, int(np.ceil(sigma)))),
    }
    if any(int(s.sum()) < 4 for s in supports.values()):
        return {}

    target_area = max(float(target.sum()), 1.0)
    target_alpha = np.zeros((h, w), dtype=np.float32)
    target_snapshots: list[np.ndarray] = []
    for dose in doses:
        wanted = dose * target_area
        ys, xs = np.where(target)
        attempts = 0
        while float((target_alpha * target).sum()) < wanted and attempts < 400:
            pick = int(rng.integers(0, len(xs)))
            centre = (int(xs[pick]), int(ys[pick]))
            add_gaussian(target_alpha, *centre, sigma)
            attempts += 1
        target_snapshots.append(target_alpha.copy())

    result = {"target": target_snapshots}
    for location in ("other_bird", "background"):
        support = supports[location]
        snapshots = []
        for target_snapshot in target_snapshots:
            control = translate_exact_mask(target_snapshot, support, rng)
            if control is None:
                return {}
            snapshots.append(control)
        result[location] = snapshots
    return result


def local_mean_colour(rgb: np.ndarray, alpha: np.ndarray) -> tuple[int, int, int]:
    selected = alpha > .05
    if not selected.any():
        values = rgb.reshape(-1, 3)
    else:
        ys, xs = np.where(selected)
        pad = 12
        y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad + 1)
        local = rgb[y0:y1, x0:x1]
        local_mask = selected[y0:y1, x0:x1]
        values = local[~local_mask]
        if not len(values):
            values = rgb.reshape(-1, 3)
    return tuple(np.median(values, axis=0).round().astype(np.uint8).tolist())


def apply_fill(image: Image.Image, alpha: np.ndarray, fill: str) -> Image.Image:
    image = image.convert("RGB")
    if fill == "local_blur":
        replacement = image.filter(ImageFilter.GaussianBlur(radius=8))
    elif fill == "local_mean":
        rgb = np.asarray(image)
        replacement = Image.new("RGB", image.size, local_mean_colour(rgb, alpha))
    else:
        raise ValueError(fill)
    mask = Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8), "L")
    return Image.composite(replacement, image, mask)


def overlay(image: Image.Image, alpha: np.ndarray, colour=(255, 30, 30)) -> Image.Image:
    layer = Image.new("RGB", image.size, colour)
    mask = Image.fromarray(np.clip(alpha * 150, 0, 150).astype(np.uint8), "L")
    return Image.composite(layer, image.convert("RGB"), mask)


def choose_items(records, dataset_name: str, max_per_part: int, min_mask_frac: float,
                 seed: int) -> tuple[list[dict], dict]:
    candidates = defaultdict(list)
    counters = defaultdict(int)
    for index, class_label, image_path, mask_source, positives in records:
        if not Path(image_path).exists():
            counters["missing_rgb"] += 1
            continue
        image = Image.open(image_path).convert("RGB")
        if dataset_name == "funnybirds":
            part_map = Path(mask_source)
            if not part_map.exists():
                counters["missing_mask_source"] += 1
                continue
            part_rgb = np.asarray(Image.open(part_map).convert("RGB").resize(
                image.size, Image.Resampling.NEAREST))
            from funnybirds_concepts import PARTMAP_COLOR_TO_INSTANCE
            bird = np.zeros(part_rgb.shape[:2], dtype=bool)
            for colour in PARTMAP_COLOR_TO_INSTANCE:
                bird |= np.all(part_rgb == np.asarray(colour, dtype=part_rgb.dtype), axis=2)
        by_part = defaultdict(list)
        for concept_idx, concept_name, part in positives:
            by_part[part].append((concept_idx, concept_name))
        for part, concepts in by_part.items():
            counters["positive_image_parts"] += 1
            if dataset_name == "funnybirds":
                target = funnybird_mask(Path(mask_source), part, image.size)
            else:
                target, bird = cub_masks(Path(mask_source), class_label,
                                         image_path.stem, part, image.size)
            if float(target.mean()) < min_mask_frac:
                counters["mask_too_small"] += 1
                continue
            key = stable_seed(seed, index, part, 0, "sample")
            candidates[part].append((key, {
                "image_index": index, "class_label": class_label,
                "image_path": Path(image_path), "image": image,
                "part": part, "concepts": concepts,
                "target": target, "bird": bird,
            }))
    selected = []
    for part, values in candidates.items():
        values.sort(key=lambda pair: pair[0])
        selected.extend(item for _, item in values[:max_per_part])
        counters[f"eligible_{part}"] = len(values)
        counters[f"selected_{part}"] = min(len(values), max_per_part)
    return selected, dict(counters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["funnybirds", "cub70"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--funnybirds-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--doses", default="0.08,0.18,0.35,0.55")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--sigma-px", type=float, default=4.0)
    ap.add_argument("--max-image-parts-per-part", type=int, default=100)
    ap.add_argument("--min-mask-frac", type=float, default=.001)
    ap.add_argument("--examples-per-part", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use an already allocated GPU session.")
    doses = sorted(float(value) for value in args.doses.split(","))
    if not doses or doses[0] <= 0 or doses[-1] >= .8:
        raise ValueError("doses must be inside (0, 0.8); this is not whole-part deletion")
    if args.repeats < 2:
        raise ValueError("at least two randomized placements are required")

    data_root = Path(args.data_root)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    example_dir = out.parent / f"{out.stem}_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, n_concepts, dataset, cfg = model_and_dataset(
        args.config, args.seed, args.epoch, device)
    if args.dataset == "funnybirds":
        root = Path(args.funnybirds_root or data_root / "FunnyBirds")
        records = funnybird_records(root)
        size = int(cfg["data"].get("img_size", 224))
        transform = T.Compose([
            T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
    else:
        records = cub_records(dataset, cfg, args.seed, data_root)
        transform = dataset.transform

    items, counters = choose_items(records, args.dataset,
                                   args.max_image_parts_per_part,
                                   args.min_mask_frac, args.seed)
    rows, examples = [], defaultdict(int)
    for item in tqdm(items, desc=f"{args.dataset} randomized masks"):
        image = item["image"]
        original_z = score(model, transform, [image], n_concepts, device)[0]
        for repeat in range(args.repeats):
            # Keep patches small relative to tiny parts such as eyes, while
            # retaining the preregistered upper bound for larger regions.
            local_sigma = min(args.sigma_px,
                              max(1.25, float(np.sqrt(item["target"].sum())) * .10))
            rng = np.random.default_rng(stable_seed(
                args.seed, item["image_index"], item["part"], repeat, "masks"))
            masks = make_patch_masks(item["target"], item["bird"], doses,
                                     local_sigma, rng)
            if not masks:
                counters["no_control_support"] = counters.get("no_control_support", 0) + 1
                continue
            for dose_index in range(len(doses)):
                masses = [float(masks[location][dose_index].sum())
                          for location in LOCATIONS]
                if max(masses) - min(masses) > max(1e-5 * max(masses), 1e-5):
                    raise RuntimeError("translated control does not preserve alpha mass")
            variants, metadata = [], []
            for fill in FILLS:
                for location in LOCATIONS:
                    for requested_dose, alpha in zip(doses, masks[location]):
                        variants.append(apply_fill(image, alpha, fill))
                        metadata.append((fill, location, requested_dose, alpha))
            masked_z = score(model, transform, variants, n_concepts, device)
            for variant, values, meta in zip(variants, masked_z, metadata):
                fill, location, requested_dose, alpha = meta
                alpha_mass = float(alpha.sum())
                target_mass = float((alpha * item["target"]).sum())
                rgb0 = np.asarray(image, dtype=np.float32)
                rgb1 = np.asarray(variant, dtype=np.float32)
                rgb_abs = np.abs(rgb1 - rgb0)
                for concept_idx, concept_name in item["concepts"]:
                    z0 = float(original_z[concept_idx]); z1 = float(values[concept_idx])
                    p0 = float(torch.sigmoid(torch.tensor(z0)))
                    p1 = float(torch.sigmoid(torch.tensor(z1)))
                    rows.append({
                        "dataset": args.dataset, "config": args.config,
                        "seed": args.seed, "epoch": args.epoch,
                        "image_index": item["image_index"],
                        "image": item["image_path"].stem,
                        "class_label": item["class_label"], "part": item["part"],
                        "concept_idx": int(concept_idx), "concept_name": concept_name,
                        "repeat": repeat, "fill": fill, "location": location,
                        "requested_dose": requested_dose,
                        "patch_sigma_px": local_sigma,
                        "mask_alpha_mass": alpha_mass,
                        "target_coverage": target_mass / max(float(item["target"].sum()), 1),
                        "damage_per_target_area": alpha_mass / max(float(item["target"].sum()), 1),
                        "z_original": z0, "z_masked": z1, "delta_z": z1 - z0,
                        "p_original": p0, "p_masked": p1, "drop_p": p0 - p1,
                        "score_still_positive": bool(z1 > 0),
                        "rgb_mae": float(rgb_abs.mean()),
                        "rgb_changed_fraction": float((rgb_abs.max(axis=2) > 0).mean()),
                        "original_sha256": sha256_image(image),
                        "masked_sha256": sha256_image(variant),
                    })
            if examples[item["part"]] < args.examples_per_part and repeat == 0:
                panels, labels = [image], ["original"]
                for location, colour in [("target", (255, 30, 30)),
                                         ("other_bird", (0, 110, 255)),
                                         ("background", (30, 160, 80))]:
                    alpha = masks[location][-1]
                    panels.extend([overlay(image, alpha, colour),
                                   apply_fill(image, alpha, "local_blur"),
                                   apply_fill(image, alpha, "local_mean")])
                    labels.extend([f"{location} mask", "blur", "local mean"])
                width, height = image.size
                sheet = Image.new("RGB", (width * len(panels), height + 28), "white")
                draw = ImageDraw.Draw(sheet)
                for col, (label, panel) in enumerate(zip(labels, panels)):
                    draw.text((col * width + 3, 5), label, fill="black")
                    sheet.paste(panel, (col * width, 28))
                sheet.save(example_dir / f"{item['part']}_{item['image_index']:06d}.png")
                examples[item["part"]] += 1

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no randomized masks were evaluated")
    # Preserve the expensive forward-pass output before applying the no-op gate.
    # A local blur on an almost uniform rendered surface can genuinely change no
    # RGB byte. That is not evidence of model insensitivity: no intervention
    # happened. Drop the complete matched target/control/background unit rather
    # than keeping one side or aborting after all inference has completed.
    raw_out = out.with_name(out.stem + ".all_rows.parquet")
    frame.to_parquet(raw_out, index=False)
    expected_parts = set(frame.part.unique())
    frame["rgb_changed"] = frame.original_sha256 != frame.masked_sha256
    unit = ["image_index", "part", "repeat", "fill", "requested_dose"]
    required = frame[frame.location.isin(["target", "other_bird"])]
    validity = (required.groupby(unit + ["location"]).rgb_changed.all()
                .unstack("location"))
    for location in ["target", "other_bird"]:
        if location not in validity:
            validity[location] = False
    valid_units = validity[validity.target & validity.other_bird].reset_index()[unit]
    total_units = int(frame[unit].drop_duplicates().shape[0])
    frame = frame.merge(valid_units, on=unit, how="inner", validate="many_to_one")
    kept_units = int(frame[unit].drop_duplicates().shape[0])
    counters["no_op_matched_units_dropped"] = total_units - kept_units
    no_op_counts = (required.loc[~required.rgb_changed]
                    .groupby(["fill", "location"]).size()
                    .rename("rows").reset_index().to_dict("records"))
    if frame.empty:
        raise RuntimeError("every matched target/control unit contained a no-op edit")
    coverage = (frame.groupby(["part", "fill"]).agg(
        units=("image_index", "size"),
        doses=("requested_dose", "nunique")).reset_index())
    expected_pairs = {(part, fill) for part in expected_parts for fill in FILLS}
    observed_pairs = set(zip(coverage.part, coverage.fill))
    if expected_pairs != observed_pairs or (coverage.doses < 3).any():
        raise RuntimeError(
            "no-op filtering removed a part/fill or left fewer than three doses; "
            f"coverage={coverage.to_dict('records')}")
    frame.to_parquet(out, index=False)
    audit = {
        "status": "PASS", "dataset": args.dataset, "config": args.config,
        "seed": args.seed, "epoch": args.epoch, "rows": len(frame),
        "images": int(frame.image.nunique()), "parts": sorted(frame.part.unique()),
        "doses": doses, "repeats": args.repeats, "sigma_px": args.sigma_px,
        "fills": list(FILLS), "locations": list(LOCATIONS),
        "max_image_parts_per_part": args.max_image_parts_per_part,
        "selection_counts": counters, "examples": str(example_dir),
        "no_op_rows_by_fill_and_location": no_op_counts,
        "raw_pre_gate_rows": len(rows), "raw_pre_gate_path": str(raw_out),
        "matched_units_before_no_op_gate": total_units,
        "matched_units_after_no_op_gate": kept_units,
        "post_gate_coverage": coverage.to_dict("records"),
        "claim_limit": "local pixel reliance and partial-context retention only; not a renderer-quality swap",
    }
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[RANDOMIZED PATCH MASKING PASS] {len(frame)} rows -> {out}")


if __name__ == "__main__":
    main()
