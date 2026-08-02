#!/usr/bin/env python3
"""Run the same raw-z mask-deletion experiment on FunnyBirds or CUB70.

The scientific unit is one positive exact concept on one image whose mapped
part is visibly annotated.  Four versions of that *same* image are scored:

  original, target part deleted, same-shape control region deleted, part only.

The primary value is target_minus_control_z.  It asks whether deleting the
named part hurts its own exact concept more than equally shaped image damage.
All component scores are retained; the adjusted difference is never interpreted
without them.  This script does not train and does not submit Slurm jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
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

from cub70_export_eval import selected_concepts
from cub70_parts import COARSE_TO_CUB70, attribute_to_part
from funnybirds_concepts import (
    PARTMAP_COLOR_TO_INSTANCE, INSTANCE_TO_COARSE, build_part_lookup,
    concept_names, group_slices, load_parts, params_to_concept_vector,
)


def sha256_image(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def read_binary_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=bool)
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def funnybird_mask(path: Path, part: str, size: tuple[int, int]) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB").resize(size, Image.Resampling.NEAREST))
    result = np.zeros(rgb.shape[:2], dtype=bool)
    for color, instance in PARTMAP_COLOR_TO_INSTANCE.items():
        if INSTANCE_TO_COARSE[instance] == part:
            result |= np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=2)
    return result


def translate_control(target: np.ndarray, bird: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Translate the target shape to a non-overlapping bird region deterministically."""
    h, w = target.shape
    ys, xs = np.where(target)
    if not len(xs):
        return np.zeros_like(target), "empty", 0.0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    candidates = []
    for dy in np.linspace(-h * .65, h * .65, 11).round().astype(int):
        for dx in np.linspace(-w * .65, w * .65, 11).round().astype(int):
            if dx == 0 and dy == 0:
                continue
            if x0 + dx < 0 or x1 + dx >= w or y0 + dy < 0 or y1 + dy >= h:
                continue
            shifted = np.zeros_like(target)
            shifted[ys + dy, xs + dx] = True
            overlap = int((shifted & target).sum())
            on_bird = int((shifted & bird & ~target).sum())
            bird_fraction = on_bird / max(int(target.sum()), 1)
            if overlap == 0 and bird_fraction >= .70:
                candidates.append((-on_bird, abs(dx) + abs(dy), shifted, bird_fraction))
    if candidates:
        candidates.sort(key=lambda item: item[:2])
        chosen = candidates[0]
        return chosen[2], "translated_same_shape_on_bird", float(chosen[3])
    # Do not silently substitute a different-shaped control. That would change
    # the question and make the two datasets less comparable.
    return np.zeros_like(target), "unavailable", 0.0


def delete_region(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, str]:
    """Remove local texture with deterministic inpainting; blur is a fallback."""
    rgb = np.asarray(image.convert("RGB"))
    try:
        import cv2
        out = cv2.inpaint(rgb, (mask.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
        return Image.fromarray(out), "telea_inpaint"
    except Exception:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=14))
        return Image.composite(blurred, image, Image.fromarray(mask.astype(np.uint8) * 255)), "blur"


def part_only(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    mean_rgb = tuple(np.asarray(rgb.mean(axis=(0, 1))).round().astype(np.uint8).tolist())
    background = Image.new("RGB", image.size, mean_rgb)
    return Image.composite(image, background, Image.fromarray(mask.astype(np.uint8) * 255))


def model_and_dataset(config: str, seed: int, epoch: int, device):
    from src.helpers import read_config
    from src.models import get_model
    bits = config.split("-")
    subpath = "-".join(bits[:2]) if len(bits) > 1 and bits[1] == "all" else bits[0]
    cfg = read_config(str(MCBM / "configs" / subpath / config))
    if config.startswith("funnybirds"):
        from grounding_deletion import load_model
        model, n_concepts = load_model(config, seed, epoch, device)
        return model, n_concepts, None, cfg
    from src.datasets import get_loader
    data_cfg = dict(cfg["data"])
    data_cfg["batch_size"] = 1
    loader, kwargs, _ = get_loader(train=False, seed=seed,
                                    return_nuisances=True, **data_cfg)
    model = get_model(**kwargs, **cfg["model"])
    ckpt = MCBM / "results" / config / str(seed) / "models" / f"epoch_{epoch}.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"[model] loaded {ckpt}")
    return model, kwargs["n_concepts"], loader.dataset, cfg


@torch.inference_mode()
def score(model, transform, images: list[Image.Image], n_concepts: int, device) -> np.ndarray:
    batch = torch.stack([transform(image) for image in images]).to(device)
    dummy = torch.zeros(len(images), n_concepts, device=device)
    out = model(batch, dummy)
    return out["c_logits"].reshape(len(images), n_concepts).float().cpu().numpy()


def funnybird_records(root: Path):
    parts = load_parts(root)
    lookup = build_part_lookup(parts)
    spans = group_slices(parts)
    names = concept_names(parts)
    params = json.loads((root / "dataset_test.json").read_text())
    rows = []
    for index, entry in enumerate(params):
        label = np.asarray(params_to_concept_vector(parts, lookup, entry))
        image_path = root / "test" / str(entry["class_idx"]) / f"{index:06d}.png"
        map_path = root / "test_part_map" / str(entry["class_idx"]) / f"{index:06d}.png"
        positives = []
        for part, (start, stop) in spans.items():
            for concept_idx in np.flatnonzero(label[start:stop]) + start:
                positives.append((int(concept_idx), names[concept_idx], part))
        rows.append((index, int(entry["class_idx"]), image_path, map_path, positives))
    return rows


def cub_records(dataset, cfg, seed: int, data_root: Path):
    names = selected_concepts(Path(cfg["data"]["attr_dir"]),
                              int(cfg["data"]["n_groups_concepts"]), seed)
    records = []
    archive = data_root / "cub70" / "masks" / "AnnotationMasksPerclass"
    for index, record in enumerate(dataset.data):
        label = np.asarray(record["attribute_label"])[dataset.concepts_idxs]
        positives = []
        for concept_idx in np.flatnonzero(label):
            name = names[int(concept_idx)]
            part = attribute_to_part(name)
            if part:
                positives.append((int(concept_idx), name, part))
        image_path = Path(record["img_path"])
        records.append((index, int(record["class_label"]), image_path, archive, positives))
    return records


def cub_masks(archive: Path, class_label: int, stem: str, part: str,
              size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    class_dir = archive / str(class_label + 1)
    target = np.zeros((size[1], size[0]), dtype=bool)
    bird = np.zeros_like(target)
    for coarse, fine_parts in COARSE_TO_CUB70.items():
        combined = np.zeros_like(target)
        for fine in fine_parts:
            combined |= read_binary_mask(class_dir / f"{stem}_{fine}.png", size)
        bird |= combined
        if coarse == part:
            target = combined
    return target, bird


def overlay(image: Image.Image, mask: np.ndarray, color=(255, 0, 0)) -> Image.Image:
    base = image.convert("RGB").copy()
    layer = Image.new("RGB", base.size, color)
    alpha = Image.fromarray(mask.astype(np.uint8) * 110)
    return Image.composite(layer, base, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["funnybirds", "cub70"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--data-root", default=os.environ.get("CURATED_DATA", ""))
    ap.add_argument("--funnybirds-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mask-frac", type=float, default=.001)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples-per-part", type=int, default=2)
    ap.add_argument("--batch-parts", type=int, default=16,
                    help="image-part interventions per GPU forward pass (4 images each)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    example_dir = out.with_suffix("").parent / (out.stem + "_examples")
    example_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this inside an allocated GPU session; "
            "do not launch the full suite on a login node.")
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
    if args.limit:
        records = records[:args.limit]

    rows, pending, example_counts = [], [], defaultdict(int)
    counters = defaultdict(int)

    def flush_pending() -> None:
        if not pending:
            return
        flat_images = [image for item in pending for image in item["images"]]
        all_scores = score(model, transform, flat_images, n_concepts, device)
        all_scores = all_scores.reshape(len(pending), 4, n_concepts)
        for item, intervention_z in zip(pending, all_scores):
            for concept_idx, concept_name in item["concepts"]:
                z_orig, z_target, z_control, z_part_only = map(
                    float, intervention_z[:, concept_idx])
                rows.append({
                    **item["metadata"], "concept_idx": concept_idx,
                    "concept_name": concept_name,
                    "z_original": z_orig, "z_target_deleted": z_target,
                    "z_control_deleted": z_control, "z_part_only": z_part_only,
                    "target_delta_z": z_target - z_orig,
                    "control_delta_z": z_control - z_orig,
                    "target_minus_control_z": z_target - z_control,
                    "context_minus_part_only_z": z_target - z_part_only,
                    "context_retains_positive_z": bool(z_target > 0),
                    "part_only_positive_z": bool(z_part_only > 0),
                })
        pending.clear()

    for index, class_label, image_path, mask_source, positives in tqdm(
            records, desc=f"{args.dataset} shared deletion"):
        counters["records_seen"] += 1
        if not image_path.exists():
            counters["missing_rgb"] += 1
            continue
        image = Image.open(image_path).convert("RGB")
        if args.dataset == "funnybirds":
            part_map = Path(mask_source)
            if not part_map.exists():
                counters["missing_mask_source"] += 1
                continue
            all_rgb = np.asarray(Image.open(part_map).convert("RGB").resize(
                image.size, Image.Resampling.NEAREST))
            bird = np.zeros(all_rgb.shape[:2], dtype=bool)
            for color in PARTMAP_COLOR_TO_INSTANCE:
                bird |= np.all(all_rgb == np.asarray(color, dtype=all_rgb.dtype), axis=2)
        positives_by_part = defaultdict(list)
        for concept_idx, concept_name, part in positives:
            positives_by_part[part].append((concept_idx, concept_name))
        for part, part_concepts in positives_by_part.items():
            counters["positive_image_parts"] += 1
            if args.dataset == "funnybirds":
                target = funnybird_mask(Path(mask_source), part, image.size)
            else:
                target, bird = cub_masks(Path(mask_source), class_label,
                                         image_path.stem, part, image.size)
            fraction = float(target.mean())
            if fraction < args.min_mask_frac:
                counters["mask_too_small"] += 1
                continue
            control, control_kind, control_bird_fraction = translate_control(target, bird)
            if not control.any():
                counters["no_same_shape_control"] += 1
                continue
            counters["evaluated_image_parts"] += 1
            target_deleted, method = delete_region(image, target)
            control_deleted, control_method = delete_region(image, control)
            isolated = part_only(image, target)
            pending.append({
                "images": [image, target_deleted, control_deleted, isolated],
                "concepts": part_concepts,
                "metadata": {
                    "dataset": args.dataset, "config": args.config, "seed": args.seed,
                    "epoch": args.epoch, "image_index": index, "image": image_path.stem,
                    "class_label": class_label, "part": part,
                    "mask_pixels": int(target.sum()), "mask_fraction": fraction,
                    "control_pixels": int(control.sum()), "control_kind": control_kind,
                    "control_bird_fraction": control_bird_fraction,
                    "deletion_method": method, "control_method": control_method,
                    "original_sha256": sha256_image(image),
                    "target_deleted_sha256": sha256_image(target_deleted),
                    "control_deleted_sha256": sha256_image(control_deleted),
                },
            })
            if len(pending) >= args.batch_parts:
                flush_pending()
            if example_counts[part] < args.examples_per_part:
                panels = [image, overlay(image, target), target_deleted,
                          overlay(image, control, (0, 100, 255)), control_deleted, isolated]
                labels = ["original", "target mask", "target deleted",
                          "control mask", "control deleted", "part only"]
                sheet = Image.new("RGB", (image.width * 6, image.height + 24), "white")
                draw = ImageDraw.Draw(sheet)
                for col, (label, panel) in enumerate(zip(labels, panels)):
                    draw.text((col * image.width + 3, 4), label, fill="black")
                    sheet.paste(panel, (col * image.width, 24))
                sheet.save(example_dir / f"{part}_{index:06d}.png")
                example_counts[part] += 1

    flush_pending()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no eligible visible positive concepts were evaluated")
    if not np.isfinite(frame[["z_original", "z_target_deleted", "z_control_deleted",
                              "z_part_only", "target_minus_control_z"]]).all().all():
        raise RuntimeError("non-finite intervention score")
    if not (frame.original_sha256 != frame.target_deleted_sha256).all():
        raise RuntimeError("at least one target deletion did not change the image")
    if not (frame.original_sha256 != frame.control_deleted_sha256).all():
        raise RuntimeError("at least one control deletion did not change the image")
    frame.to_parquet(out, index=False)
    summary = (frame.groupby("part").agg(
        n=("image", "size"), n_images=("image", "nunique"),
        median_target_delta_z=("target_delta_z", "median"),
        median_control_delta_z=("control_delta_z", "median"),
        median_target_minus_control_z=("target_minus_control_z", "median"),
        median_context_minus_part_only_z=("context_minus_part_only_z", "median"),
        context_positive_rate=("context_retains_positive_z", "mean"),
        part_only_positive_rate=("part_only_positive_z", "mean"),
    ).reset_index())
    summary.to_csv(out.with_suffix(".summary.csv"), index=False)
    audit = {
        "dataset": args.dataset, "config": args.config, "seed": args.seed,
        "epoch": args.epoch, "rows": len(frame), "images": int(frame.image.nunique()),
        "parts": sorted(frame.part.unique()), "min_mask_frac": args.min_mask_frac,
        "examples": str(example_dir), "selection_counts": dict(counters),
        "part_only_method": "named-part pixels on the image's global-mean RGB field",
        "status": "PASS",
    }
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(f"[PAIRED MASK DELETION PASS] {len(frame)} rows -> {out}")


if __name__ == "__main__":
    main()
