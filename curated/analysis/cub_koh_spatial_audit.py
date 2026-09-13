#!/usr/bin/env python3
"""Audit where an accepted Koh Joint CUB concept logit is spatially sensitive.

This is frozen inference, not training.  For each selected positive concept we
compute concept-specific Grad-CAM at the ResNet-50 ``layer4`` feature map and
compare it with the released CUB70 part mask.  We also replay the unchanged
linear Koh species head after replacing within-label raw-logit magnitudes by
cross-fitted label means.  The two outputs deliberately answer different
questions: spatial localization and downstream use of magnitude information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
for path in (CURATED / "compat", CURATED / "external" / "ConceptBottleneck",
             CURATED / "data" / "cub70"):
    sys.path.insert(0, str(path))

from cub70_parts import ATTRIBUTE_TYPE_TO_MASK, COARSE_TO_CUB70  # noqa: E402
from relabel_cub_with_cub70 import coarse_visibility  # noqa: E402


def family(name: str) -> str:
    return str(name).split("::", 1)[0]


def image_stem(value: object) -> str:
    return Path(str(value)).stem


def stable_fold(value: object, folds: int = 5) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest, 16) % folds


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = values.sum()
    return values / total if total > 0 else np.zeros_like(values)


def localization_metrics(heatmap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return mask-overlap metrics for one non-negative spatial heatmap."""
    raw_heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float64), nan=0.0,
                                posinf=0.0, neginf=0.0)
    raw_heatmap = np.maximum(raw_heatmap, 0.0)
    has_signal = bool(raw_heatmap.sum() > 0)
    heatmap = normalize_map(raw_heatmap)
    mask = np.asarray(mask, dtype=bool)
    if heatmap.shape != mask.shape:
        raise ValueError(f"heatmap/mask shape mismatch: {heatmap.shape} vs {mask.shape}")
    area = float(mask.mean())
    if not mask.any() or area <= 0:
        raise ValueError("localization metrics require a non-empty mask")
    mass = float(heatmap[mask].sum())
    enrichment = mass / area
    # An all-zero positive Grad-CAM has no maximum location.  Calling pixel zero
    # a "hit" would manufacture localization evidence, so retain it as missing.
    pointing = (float(mask[np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)])
                if has_signal else np.nan)
    k = int(mask.sum())
    selected = np.zeros(mask.size, dtype=bool)
    if k and has_signal:
        selected[np.argpartition(heatmap.ravel(), -k)[-k:]] = True
    selected = selected.reshape(mask.shape)
    intersection = int((selected & mask).sum())
    union = int((selected | mask).sum())
    return {
        "mask_area_fraction": area,
        "attribution_mass_inside_mask": mass,
        "area_adjusted_enrichment": enrichment,
        "pointing_inside_mask": pointing,
        "equal_area_iou": float(intersection / union) if union else np.nan,
        "has_spatial_signal": float(has_signal),
    }


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_fitted_label_means(z: np.ndarray, c: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Replace each value by its other-fold mean for the same concept and label."""
    replacement = np.empty_like(z, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        for concept in range(z.shape[1]):
            for label in np.unique(c[test, concept]):
                reference = z[train & (c[:, concept] == label), concept]
                if not len(reference):
                    raise RuntimeError(
                        f"fold {fold} has no reference rows for concept {concept}, label {label}"
                    )
                selected = test & (c[:, concept] == label)
                replacement[selected, concept] = reference.mean()
    return replacement


def saved_head_use_table(
    evaluation: pd.DataFrame, weight: np.ndarray, bias: np.ndarray
) -> pd.DataFrame:
    """Measure use of within-label magnitudes by the unchanged Koh class head."""
    z = evaluation.pivot(index="image", columns="concept_index", values="z").sort_index(axis=1)
    c = evaluation.pivot(index="image", columns="concept_index", values="gt_label").loc[z.index, z.columns]
    image_rows = evaluation[["image", "y_true", "y_pred"]].drop_duplicates("image").set_index("image").loc[z.index]
    if weight.shape[1] != z.shape[1] or bias.shape != (weight.shape[0],):
        raise RuntimeError(f"saved-head shape {weight.shape}/{bias.shape} does not match {z.shape}")
    logits = z.to_numpy(dtype=np.float64) @ weight.T + bias
    reconstructed = logits.argmax(1)
    agreement = float((reconstructed == image_rows.y_pred.to_numpy()).mean())
    if agreement < 0.999:
        raise RuntimeError(f"saved head reconstructs only {agreement:.3%} of exported predictions")
    folds = np.array([stable_fold(value) for value in z.index])
    mean_values = cross_fitted_label_means(
        z.to_numpy(dtype=np.float64), c.to_numpy(dtype=int), folds
    )
    concept_meta = evaluation[["concept_index", "concept_name"]].drop_duplicates()
    concept_meta["mask_group"] = concept_meta.concept_name.map(family).map(ATTRIBUTE_TYPE_TO_MASK)
    groups = {"all 112": list(z.columns)}
    groups.update({
        group: concept_meta.loc[concept_meta.mask_group == group, "concept_index"].tolist()
        for group in COARSE_TO_CUB70
    })
    original_probability = softmax(logits)
    rows = []
    for group, columns in groups.items():
        columns = [int(value) for value in columns]
        if not columns:
            continue
        changed = z.to_numpy(dtype=np.float64).copy()
        changed[:, columns] = mean_values[:, columns]
        changed_logits = changed @ weight.T + bias
        changed_probability = softmax(changed_logits)
        rows.append({
            "block": group,
            "coordinates_replaced": len(columns),
            "top1_change_rate": float((changed_logits.argmax(1) != reconstructed).mean()),
            "mean_probability_mass_moved": float(
                (0.5 * np.abs(changed_probability - original_probability).sum(axis=1)).mean()
            ),
            "mean_absolute_class_logit_change": float(np.abs(changed_logits - logits).mean()),
            "reconstruction_agreement": agreement,
            "n_images": len(z),
        })
    return pd.DataFrame(rows)


def load_model(checkpoint: Path, device: torch.device):
    try:
        model = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint, map_location=device)
    if getattr(model, "curated_framework", None) != "koh_joint":
        raise RuntimeError("checkpoint is not marked as Koh Joint")
    if getattr(model, "curated_backbone", None) != "resnet50":
        raise RuntimeError("checkpoint is not marked as ResNet-50")
    if not hasattr(model, "first_model") or not hasattr(model.first_model, "layer4"):
        raise RuntimeError("checkpoint has no Koh ResNet first_model.layer4")
    if not hasattr(model, "sec_model") or not hasattr(model.sec_model, "linear"):
        raise RuntimeError("checkpoint has no Koh sec_model.linear")
    return model.to(device).eval()


def coarse_mask(mask_root: Path, class_id: int, stem: str, group: str, shape: tuple[int, int]) -> np.ndarray:
    root = mask_root / "AnnotationMasksPerclass" if (mask_root / "AnnotationMasksPerclass").is_dir() else mask_root
    result = np.zeros(shape, dtype=bool)
    for part in COARSE_TO_CUB70[group]:
        path = root / str(class_id) / f"{stem}_{part}.png"
        if not path.is_file():
            continue
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if mask.shape != shape:
            mask = np.asarray(
                Image.fromarray(mask.astype("uint8") * 255).resize(
                    (shape[1], shape[0]), Image.Resampling.NEAREST
                )
            ) > 0
        result |= mask
    return result


def choose_candidates(evaluation: pd.DataFrame, visibility: pd.DataFrame, per_group: int) -> pd.DataFrame:
    frame = evaluation.copy()
    frame["image"] = frame.image.map(image_stem)
    frame["attribute_type"] = frame.concept_name.map(family)
    frame["mask_group"] = frame.attribute_type.map(ATTRIBUTE_TYPE_TO_MASK)
    visible = coarse_visibility(visibility, threshold=0.001).rename(columns={"image_name": "image"})
    visible["image"] = visible.image.map(image_stem)
    frame = frame[frame.mask_group.notna() & (frame.gt_label == 1)].merge(
        visible[["image", "coarse", "visible"]].rename(columns={"coarse": "mask_group"}),
        on=["image", "mask_group"], how="inner", validate="many_to_one"
    )
    frame = frame[frame.visible].copy()
    frame["selection_key"] = frame.apply(
        lambda row: hashlib.sha1(
            f"{row.image}|{row.concept_index}|{row.mask_group}".encode("utf-8")
        ).hexdigest(), axis=1
    )
    frame = frame.sort_values("selection_key").groupby(
        ["mask_group", "concept_name"], group_keys=False
    ).head(4)
    frame = frame.groupby("mask_group", group_keys=False).head(per_group)
    return frame[["image", "y_true", "concept_index", "concept_name", "mask_group", "z"]].reset_index(drop=True)


def tensor_rgb(image: torch.Tensor) -> np.ndarray:
    rgb = (2.0 * image.detach().cpu() + 0.5).clamp(0, 1)
    return np.transpose(rgb.numpy(), (1, 2, 0))


def overlay(rgb: np.ndarray, values: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    values = values / values.max() if values.max() > 0 else values
    result = rgb.copy()
    tint = np.zeros_like(result) + np.asarray(color)
    alpha = (0.65 * values)[..., None]
    return np.clip(result * (1 - alpha) + tint * alpha, 0, 1)


def build_example_sheet(metrics: pd.DataFrame, out: Path) -> None:
    groups = [group for group in COARSE_TO_CUB70 if group in set(metrics.mask_group)]
    picks = []
    for group in groups:
        local = metrics[metrics.mask_group == group]
        picks.extend([
            (group, "least localized", local.nsmallest(1, "positive_area_adjusted_enrichment").iloc[0]),
            (group, "most localized", local.nlargest(1, "positive_area_adjusted_enrichment").iloc[0]),
        ])
    width, row_height = 1200, 310
    sheet = Image.new("RGB", (width, row_height * len(picks)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (group, label, record) in enumerate(picks):
        payload = np.load(record.map_path)
        rgb, mask = payload["rgb"], payload["mask"].astype(bool)
        positive, absolute = payload["positive"], payload["absolute"]
        mask_view = rgb.copy(); mask_view[mask] = 0.35 * mask_view[mask] + 0.65 * np.array([0.0, 0.65, 1.0])
        panels = [rgb, mask_view, overlay(rgb, positive, (1.0, 0.1, 0.0)), overlay(rgb, absolute, (0.7, 0.0, 0.8))]
        y0 = row_index * row_height
        draw.text((8, y0 + 5),
                  f"{group} | {label} | {record.concept_name} | z={record.z:.2f} | "
                  f"positive enrichment={record.positive_area_adjusted_enrichment:.2f}", fill="black")
        for panel_index, panel in enumerate(panels):
            image = Image.fromarray((panel * 255).astype("uint8"))
            image.thumbnail((285, 260))
            sheet.paste(image, (8 + panel_index * 298, y0 + 35))
        for panel_index, title in enumerate(["photograph", "released mask", "positive Grad-CAM", "absolute Grad-CAM"]):
            draw.text((8 + panel_index * 298, y0 + 292), title, fill="black")
    sheet.save(out)


def run_gradcam(args, model, evaluation: pd.DataFrame) -> pd.DataFrame:
    from CUB.dataset import load_data

    visibility = pd.read_parquet(args.visibility)
    candidates = choose_candidates(evaluation, visibility, args.per_group)
    if candidates.empty:
        raise RuntimeError("no positive, visibly masked concept candidates")
    records = pickle.loads(args.data_pkl.read_bytes())
    loader = load_data([args.data_pkl], use_attr=True, no_img=False, batch_size=1,
                       uncertain_label=False, n_class_attr=2, image_dir="images",
                       resampling=False)
    by_image = {name: frame for name, frame in candidates.groupby("image")}
    activation: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        activation["value"] = output
        output.retain_grad()

    handle = model.first_model.layer4.register_forward_hook(hook)
    rows = []
    map_dir = args.out_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(args.work_dir)
        for index, batch in enumerate(loader):
            images, labels, _attributes = batch
            record = records[index]
            stem = image_stem(record.get("image", record["img_path"]))
            if stem not in by_image:
                continue
            image = images.to(args.device)
            outputs = model(image)
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 113:
                raise RuntimeError(f"unexpected Koh Joint output contract: {type(outputs)} len={len(outputs) if hasattr(outputs, '__len__') else 'NA'}")
            z = torch.cat([value.reshape(1, -1) for value in outputs[1:]], dim=1)
            features = activation["value"]
            rgb = tensor_rgb(images[0])
            for position, selected in enumerate(by_image[stem].itertuples(index=False)):
                if int(labels[0]) != int(selected.y_true):
                    raise RuntimeError(
                        f"loader/export class mismatch for {stem}: {int(labels[0])} vs {selected.y_true}"
                    )
                replayed_z = float(z[0, int(selected.concept_index)].detach().cpu())
                if abs(replayed_z - float(selected.z)) > 1e-4:
                    raise RuntimeError(
                        f"loader/export raw-z mismatch for {stem}/{selected.concept_name}: "
                        f"{replayed_z} vs {selected.z}"
                    )
                model.zero_grad(set_to_none=True)
                if features.grad is not None:
                    features.grad.zero_()
                z[0, int(selected.concept_index)].backward(retain_graph=position < len(by_image[stem]) - 1)
                gradients = features.grad[0]
                weights = gradients.mean(dim=(1, 2), keepdim=True)
                signed = (weights * features[0]).sum(dim=0)
                positive = F.relu(signed)[None, None]
                absolute = signed.abs()[None, None]
                positive = F.interpolate(positive, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
                absolute = F.interpolate(absolute, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
                mask = coarse_mask(args.mask_root, int(labels[0]) + 1, stem, selected.mask_group, rgb.shape[:2])
                if mask.mean() < 0.001:
                    raise RuntimeError(f"selected visible mask became empty: {stem} {selected.mask_group}")
                positive_metrics = localization_metrics(positive, mask)
                absolute_metrics = localization_metrics(absolute, mask)
                token = hashlib.sha1(f"{stem}|{selected.concept_index}".encode()).hexdigest()[:14]
                map_path = map_dir / f"{token}.npz"
                np.savez_compressed(map_path, rgb=rgb, mask=mask, positive=positive, absolute=absolute)
                rows.append({
                    "image": stem, "class_label": int(labels[0]),
                    "concept_index": int(selected.concept_index),
                    "concept_name": selected.concept_name, "mask_group": selected.mask_group,
                    "z": float(z[0, int(selected.concept_index)].detach().cpu()),
                    "map_path": str(map_path),
                    **{f"positive_{key}": value for key, value in positive_metrics.items()},
                    **{f"absolute_{key}": value for key, value in absolute_metrics.items()},
                })
            if len(rows) and len(rows) % 50 == 0:
                print(f"Grad-CAM evaluated {len(rows)}/{len(candidates)} selected image-concept pairs", flush=True)
    finally:
        handle.remove()
        os.chdir(old_cwd)
    result = pd.DataFrame(rows)
    if len(result) != len(candidates):
        missing = set(map(tuple, candidates[["image", "concept_index"]].to_numpy())) - set(map(tuple, result[["image", "concept_index"]].to_numpy()))
        raise RuntimeError(f"evaluated {len(result)} of {len(candidates)} selected pairs; missing={list(missing)[:10]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-pkl", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cub70", "cub"), required=True)
    parser.add_argument("--per-group", type=int, default=48)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.type != "cuda":
        raise RuntimeError("concept Grad-CAM requires a CUDA allocation; it performs no training")
    manifest_path = args.model_root / "SUCCESS.json"
    checkpoint_manifest_path = args.model_root / "CHECKPOINT.json"
    for path in (manifest_path, checkpoint_manifest_path, args.data_pkl, args.visibility):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest.get("metadata", {})
    expected = {"framework": "koh_joint", "backbone": "resnet50",
                "dataset": args.dataset, "labels": "standard", "seed": "1"}
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items()
                  if str(metadata.get(key)) != str(value)}
    if mismatched:
        raise RuntimeError(f"model manifest mismatch: {mismatched}")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
    checkpoint = Path(checkpoint_manifest["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    evaluation_path = args.model_root / "final_test.parquet"
    evaluation = pd.read_parquet(evaluation_path)
    evaluation["image"] = evaluation.image.map(image_stem)
    required = {"image", "concept_index", "concept_name", "z", "gt_label", "y_true", "y_pred"}
    if missing := required - set(evaluation):
        raise RuntimeError(f"evaluation missing columns: {sorted(missing)}")
    model = load_model(checkpoint, args.device)
    head = model.sec_model.linear
    head_use = saved_head_use_table(
        evaluation,
        head.weight.detach().cpu().numpy().astype(np.float64),
        head.bias.detach().cpu().numpy().astype(np.float64),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head_use.to_csv(args.out_dir / "saved_head_use.csv", index=False)
    metrics = run_gradcam(args, model, evaluation)
    metrics.to_parquet(args.out_dir / "gradcam_metrics.parquet", index=False)
    summary = metrics.groupby("mask_group").agg(
        n_pairs=("image", "size"), n_images=("image", "nunique"),
        n_concepts=("concept_name", "nunique"),
        median_mask_area=("positive_mask_area_fraction", "median"),
        median_positive_mass_inside=("positive_attribution_mass_inside_mask", "median"),
        median_positive_enrichment=("positive_area_adjusted_enrichment", "median"),
        positive_pointing_rate=("positive_pointing_inside_mask", "mean"),
        positive_spatial_signal_rate=("positive_has_spatial_signal", "mean"),
        median_positive_equal_area_iou=("positive_equal_area_iou", "median"),
        median_absolute_mass_inside=("absolute_attribution_mass_inside_mask", "median"),
        median_absolute_enrichment=("absolute_area_adjusted_enrichment", "median"),
    ).reset_index()
    summary.to_csv(args.out_dir / "gradcam_summary.csv", index=False)
    build_example_sheet(metrics, args.out_dir / "gradcam_examples.png")
    success = {
        "status": "ACCEPTED FOR POST-HOC SPATIAL DIAGNOSTIC",
        "training": False,
        "framework": "koh_joint",
        "backbone": "resnet50",
        "dataset": args.dataset,
        "seed": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "evaluation": str(evaluation_path),
        "selected_pairs": len(metrics),
        "mask_groups": sorted(metrics.mask_group.unique()),
        "method_boundary": "concept-specific Grad-CAM is post-hoc sensitivity, not SEG-MIL-CBM exact segment contribution and not a CUB donor swap",
    }
    (args.out_dir / "SUCCESS.json").write_text(json.dumps(success, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(head_use.round(4).to_string(index=False))
    print(f"[CUB KOH SPATIAL AUDIT SUCCESS] {args.out_dir}")


if __name__ == "__main__":
    main()
