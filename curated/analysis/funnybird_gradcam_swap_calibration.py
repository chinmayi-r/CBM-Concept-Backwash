#!/usr/bin/env python3
"""Calibrate concept-specific Grad-CAM against accepted FunnyBird swaps.

The renderer swap is the accepted causal measurement.  This module asks a
narrower question: does a post-hoc localization score track that measurement?
It never treats Grad-CAM as ground truth and it never trains a new model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from cub_koh_spatial_audit import gradcam_maps, localization_metrics
from funnybird_allpart_swap_preflight import load_koh_model
import funnybirds_concepts as fbc


PARTS = ("tail", "wing", "beak", "foot", "eye")
COLORS = {
    "tail": "#6f0db7", "wing": "#0077b6", "beak": "#eea400",
    "foot": "#009e73", "eye": "#c774a5",
}
MODEL_SIZE = 299


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funnybirds-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--swaps", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows-per-part", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def stable_order(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    local = frame.copy()
    local["selection_key"] = local.render_id.map(
        lambda value: hashlib.sha1(f"{seed}|{value}".encode()).hexdigest())
    return local.sort_values("selection_key")


def select_rows(swaps: pd.DataFrame, rows_per_part: int, seed: int) -> pd.DataFrame:
    """Deterministic outcome-blind sample, round-robin across donor values."""
    if rows_per_part < 10:
        raise ValueError("--rows-per-part must be at least 10")
    local = swaps.copy()
    local["controlled_event"] = ((local.response_delta > 0) & (local.margin < 0))
    chosen = []
    for part in PARTS:
        block = local[local.part == part]
        if block.empty:
            raise RuntimeError(f"accepted swap population has no {part} rows")
        ordered = stable_order(block, seed)
        buckets = {value: group.copy() for value, group in ordered.groupby("var_donor")}
        positions = {value: 0 for value in buckets}
        part_rows, donor_values = [], sorted(buckets)
        while len(part_rows) < rows_per_part:
            added = False
            for value in donor_values:
                position = positions[value]
                if position < len(buckets[value]):
                    part_rows.append(buckets[value].iloc[position])
                    positions[value] += 1
                    added = True
                    if len(part_rows) == rows_per_part:
                        break
            if not added:
                break
        part_rows = pd.DataFrame(part_rows)
        chosen.append(part_rows.head(rows_per_part))
    selected = pd.concat(chosen, ignore_index=True)
    if selected.groupby("part").size().reindex(PARTS).min() != rows_per_part:
        raise RuntimeError("could not select the requested number of rows for every part")
    return selected


def center_crop_array(values: np.ndarray, size: int = MODEL_SIZE) -> np.ndarray:
    """Mirror torchvision CenterCrop, including zero padding for small inputs."""
    values = np.asarray(values)
    if values.ndim not in (2, 3):
        raise ValueError(f"expected HxW or HxWxC array, got {values.shape}")
    height, width = values.shape[:2]
    pad_left = max((size - width) // 2, 0)
    pad_right = max(size - width - pad_left, 0)
    pad_top = max((size - height) // 2, 0)
    pad_bottom = max(size - height - pad_top, 0)
    pads = ((pad_top, pad_bottom), (pad_left, pad_right))
    if values.ndim == 3:
        pads += ((0, 0),)
    padded = np.pad(values, pads, mode="constant")
    height, width = padded.shape[:2]
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return padded[top:top + size, left:left + size]


def changed_pixel_mask(original: Image.Image, counterfactual: Image.Image) -> np.ndarray:
    original_rgb = np.asarray(original.convert("RGB"))
    counterfactual_rgb = np.asarray(counterfactual.convert("RGB"))
    if original_rgb.shape != counterfactual_rgb.shape:
        raise RuntimeError(
            f"accepted swap RGB shapes differ: {original_rgb.shape} vs {counterfactual_rgb.shape}")
    native = np.any(original_rgb != counterfactual_rgb, axis=2)
    mask = center_crop_array(native).astype(bool)
    if not mask.any():
        raise RuntimeError("accepted swap has no changed pixels in the model view")
    return mask


def model_tensor(image: Image.Image) -> torch.Tensor:
    rgb = center_crop_array(np.asarray(image.convert("RGB"))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    return (tensor - 0.5) / 2.0


def concept_vector(outputs) -> torch.Tensor:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 27:
        raise RuntimeError("expected Koh class output plus 26 scalar concept outputs")
    values = torch.cat([item.reshape(item.shape[0], -1) for item in outputs[1:]], dim=1)
    if values.shape[1] != 26:
        raise RuntimeError(f"expected 26 concept logits, got {tuple(values.shape)}")
    return values


def overlay(rgb: np.ndarray, values: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    values = values / values.max() if values.max() > 0 else values
    alpha = (0.65 * values)[..., None]
    tint = np.zeros_like(rgb) + np.asarray(color)
    return np.clip(rgb * (1 - alpha) + tint * alpha, 0, 1)


def run_gradcam(model, selected: pd.DataFrame, device: torch.device,
                spans: dict[str, tuple[int, int]]) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    activation: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        activation["value"] = output
        output.retain_grad()

    handle = model.first_model.layer4.register_forward_hook(hook)
    rows = []
    extremes: dict[str, dict[str, object]] = {}
    try:
        for number, row in enumerate(selected.itertuples(index=False), 1):
            original = Image.open(row.image_orig_path).convert("RGB")
            counterfactual = Image.open(row.image_cf_path).convert("RGB")
            mask = changed_pixel_mask(original, counterfactual)
            image = model_tensor(counterfactual).unsqueeze(0).to(device)
            outputs = model(image)
            z = concept_vector(outputs)
            features = activation["value"]
            lo, hi = spans[row.part]
            if int(row.var_src) >= hi - lo or int(row.var_donor) >= hi - lo:
                raise RuntimeError(
                    f"{row.render_id}: value index outside {row.part} width {hi-lo}")
            source_index = lo + int(row.var_src)
            donor_index = lo + int(row.var_donor)
            replay_source = float(z[0, source_index].detach().cpu())
            replay_donor = float(z[0, donor_index].detach().cpu())
            replay_error = max(abs(replay_source - float(row.z_old)),
                               abs(replay_donor - float(row.z_new)))
            if replay_error > 0.05:
                raise RuntimeError(
                    f"{row.render_id}: replay error {replay_error:.5f} exceeds 0.05 cap")
            targets = {
                "donor_logit": z[0, donor_index],
                "donor_minus_source_margin": z[0, donor_index] - z[0, source_index],
            }
            rgb = center_crop_array(np.asarray(counterfactual)).astype(np.float32) / 255.0
            payload = {"rgb": rgb, "mask": mask}
            record = {
                "render_id": row.render_id, "part": row.part,
                "var_src": int(row.var_src), "var_donor": int(row.var_donor),
                "orig_render_id": row.orig_render_id,
                "margin": float(row.margin), "response_delta": float(row.response_delta),
                "controlled_event": bool(row.controlled_event),
                "changed_pixel_fraction": float(mask.mean()),
                "replay_error": replay_error,
            }
            for target_name, target in targets.items():
                model.zero_grad(set_to_none=True)
                if features.grad is not None:
                    features.grad.zero_()
                target.backward(retain_graph=True)
                positive, absolute = gradcam_maps(
                    features[0], features.grad[0], mask.shape)
                payload[f"{target_name}_positive"] = positive
                payload[f"{target_name}_absolute"] = absolute
                for map_name, values in (("positive", positive), ("absolute", absolute)):
                    metrics = localization_metrics(values, mask)
                    record.update({
                        f"{target_name}_{map_name}_{key}": value
                        for key, value in metrics.items()
                    })
            rows.append(record)
            score = record[
                "donor_minus_source_margin_positive_area_adjusted_enrichment"]
            for label, better in (("least", lambda x, y: x < y),
                                  ("most", lambda x, y: x > y)):
                key = f"{row.part}:{label}"
                if key not in extremes or better(score, extremes[key]["score"]):
                    extremes[key] = {
                        "score": score, "record": record.copy(),
                        "payload": {name: np.asarray(value).copy()
                                    for name, value in payload.items()},
                    }
            if number % 25 == 0:
                print(f"FunnyBird Grad-CAM {number}/{len(selected)}", flush=True)
    finally:
        handle.remove()
    return pd.DataFrame(rows), extremes


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for part, frame in metrics.groupby("part", sort=False):
        event = frame.controlled_event.astype(int)
        enrichment = frame.donor_minus_source_margin_positive_area_adjusted_enrichment
        rows.append({
            "part": part, "n_rows": len(frame),
            "n_originals": frame.orig_render_id.nunique(),
            "controlled_event_rate": float(event.mean()),
            "median_margin_gradcam_enrichment": float(enrichment.median()),
            "median_enrichment_event": float(enrichment[event == 1].median()) if event.any() else np.nan,
            "median_enrichment_no_event": float(enrichment[event == 0].median()) if (~event.astype(bool)).any() else np.nan,
            "spearman_enrichment_response": float(enrichment.corr(frame.response_delta, method="spearman")),
            "median_changed_pixel_fraction": float(frame.changed_pixel_fraction.median()),
        })
    return pd.DataFrame(rows).set_index("part").reindex(PARTS).reset_index()


def build_examples(extremes: dict[str, dict[str, object]], out: Path) -> None:
    picks = []
    for part in PARTS:
        for key, label in ((f"{part}:least", "least localized"),
                           (f"{part}:most", "most localized")):
            item = extremes[key]
            picks.append((part, label, item["record"], item["payload"]))
    width, row_height = 1200, 310
    sheet = Image.new("RGB", (width, row_height * len(picks)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_number, (part, label, record, data) in enumerate(picks):
        rgb, mask = data["rgb"], data["mask"].astype(bool)
        mask_view = rgb.copy()
        mask_view[mask] = 0.35 * mask_view[mask] + 0.65 * np.array([0.0, 0.65, 1.0])
        donor = data["donor_logit_positive"]
        margin = data["donor_minus_source_margin_positive"]
        panels = [rgb, mask_view, overlay(rgb, donor, (1.0, 0.1, 0.0)),
                  overlay(rgb, margin, (0.7, 0.0, 0.8))]
        y = row_number * row_height
        draw.text((8, y + 5),
                  f"{part} | {label} | old {record['var_src']} -> donor {record['var_donor']} | "
                  f"response={record['response_delta']:.2f} | final margin={record['margin']:.2f} | "
                  f"backwash event={bool(record['controlled_event'])}", fill="black")
        for column, panel in enumerate(panels):
            image = Image.fromarray((panel * 255).astype("uint8"))
            image.thumbnail((285, 260))
            sheet.paste(image, (8 + column * 298, y + 35))
        for column, title in enumerate(("swapped image", "exact changed pixels",
                                        "donor-score Grad-CAM", "margin Grad-CAM")):
            draw.text((8 + column * 298, y + 292), title, fill="black")
    sheet.save(out)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    swap_path = Path(args.swaps).resolve()
    funnybirds = Path(args.funnybirds_root).resolve()
    out = Path(args.out_dir).resolve()
    for path in (checkpoint, swap_path, funnybirds / "parts.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to mix outputs in non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    swaps = pd.read_csv(swap_path)
    required = {"render_id", "orig_render_id", "part", "var_src", "var_donor",
                "image_orig_path", "image_cf_path", "image_orig_sha256", "image_cf_sha256",
                "z_old_orig", "z_new_orig", "z_old", "z_new", "margin", "response_delta"}
    missing = required - set(swaps)
    if missing:
        raise RuntimeError(f"accepted swap CSV lacks {sorted(missing)}; columns={list(swaps)}")
    if len(swaps) != 5000 or swaps.orig_render_id.nunique() != 250:
        raise RuntimeError("calibration requires the accepted 5,000 swaps and 250 originals")
    if set(swaps.part) != set(PARTS):
        raise RuntimeError(f"unexpected part population: {sorted(swaps.part.unique())}")
    margin_error = np.abs(swaps.margin - (swaps.z_new - swaps.z_old)).max()
    response_error = np.abs(
        swaps.response_delta - (swaps.margin - (swaps.z_new_orig - swaps.z_old_orig))
    ).max()
    if margin_error > 1e-6 or response_error > 1e-6:
        raise RuntimeError(
            f"accepted swap formula closure failed: margin={margin_error}, response={response_error}")
    selected = select_rows(swaps, args.rows_per_part, args.seed)
    for row in selected.itertuples(index=False):
        for path_value, expected in ((row.image_orig_path, row.image_orig_sha256),
                                     (row.image_cf_path, row.image_cf_sha256)):
            path = Path(path_value)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != str(expected):
                raise RuntimeError(f"accepted RGB hash mismatch: {path}")
    selected.to_csv(out / "selected_rows.csv", index=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Grad-CAM calibration requires CUDA; no training is performed")
    device = torch.device("cuda")
    model = load_koh_model(checkpoint, device)
    spans = fbc.group_slices(fbc.load_parts(funnybirds))
    if set(spans) != set(PARTS) or sum(hi-lo for lo,hi in spans.values()) != 26:
        raise RuntimeError(f"unexpected FunnyBird concept schema: {spans}")
    metrics, extremes = run_gradcam(model, selected, device, spans)
    numeric = metrics.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.drop(columns=[c for c in numeric if numeric[c].isna().any()])).all().all():
        raise RuntimeError("non-finite values outside explicitly undefined localization statistics")
    metrics.to_parquet(out / "gradcam_metrics.parquet", index=False)
    summary = summarize(metrics)
    summary.to_csv(out / "gradcam_swap_calibration.csv", index=False)
    build_examples(extremes, out / "gradcam_swap_examples.png")
    success = {
        "status": "CALIBRATION COMPUTED; SCIENTIFIC VERDICT REQUIRES OUTPUT REVIEW",
        "training": False, "framework": "koh_joint", "backbone": "resnet50",
        "dataset": "funnybirds", "seed": 1, "rows": len(metrics),
        "originals": int(metrics.orig_render_id.nunique()),
        "parts": list(PARTS),
        "causal_reference": "accepted renderer swaps; controlled_event = response_delta > 0 and final margin < 0",
        "spatial_mask": "pixels that differ between accepted original and counterfactual RGBs, after the model's CenterCrop(299)",
        "gradcam_targets": ["donor raw concept logit", "donor-minus-source raw concept margin"],
        "method_boundary": "Grad-CAM is calibrated against the causal swap; it is not itself declared causal",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "SUCCESS.json").write_text(json.dumps(success, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(f"[FUNNYBIRD GRADCAM/SWAP CALIBRATION COMPUTED] {out}")


if __name__ == "__main__":
    main()
