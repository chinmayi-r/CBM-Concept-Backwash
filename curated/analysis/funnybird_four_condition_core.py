"""Dependency-light calculations for the FunnyBird four-condition diagnostic."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def target_rgb_mae(a: Image.Image, b: Image.Image,
                   mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    common = np.logical_and(mask_a, mask_b)
    if not common.any():
        return float("nan")
    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.abs(aa[common] - bb[common]).mean())


def image_rgb_mae(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute 8-bit RGB difference over the complete image."""
    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB"), dtype=np.float32)
    if aa.shape != bb.shape:
        raise ValueError(f"image shapes differ: {aa.shape} versus {bb.shape}")
    return float(np.abs(aa - bb).mean())


def exact_target_composite(target_source: Image.Image, context_free: Image.Image,
                           target: np.ndarray) -> Image.Image:
    """Copy byte-identical target pixels onto one target-absent base image."""
    alpha = Image.fromarray((target.astype(np.uint8) * 255), mode="L")
    return Image.composite(target_source.convert("RGB"), context_free.convert("RGB"), alpha)


def delete_part(ann: dict, part: str, parts_with_color: set[str]) -> dict:
    result = dict(ann)
    result[f"{part}_model"] = ""
    if part in parts_with_color:
        result[f"{part}_color"] = ""
    return result


def four_conditions(ann: dict, target: str, part_names: Iterable[str],
                    parts_with_color: set[str]) -> dict[str, dict]:
    z11 = dict(ann)
    z01 = delete_part(z11, target, parts_with_color)
    z10 = dict(z11)
    for part in part_names:
        if part != target:
            z10 = delete_part(z10, part, parts_with_color)
    z00 = delete_part(z10, target, parts_with_color)
    return {"11": z11, "01": z01, "10": z10, "00": z00}


def add_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["part_response_with_context"] = result.z11 - result.z01
    result["part_response_without_context"] = result.z10 - result.z00
    result["context_evidence_with_part"] = result.z11 - result.z10
    result["context_evidence_without_part"] = result.z01 - result.z00
    result["interaction"] = (
        result.part_response_with_context - result.part_response_without_context)
    closure = (
        result.context_evidence_with_part - result.context_evidence_without_part)
    if not np.allclose(result.interaction, closure, atol=1e-7, rtol=0):
        raise RuntimeError("four-condition interaction identity did not close")
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "z11", "z01", "z10", "z00",
        "part_response_with_context", "part_response_without_context",
        "context_evidence_with_part", "context_evidence_without_part",
        "interaction",
    ]
    rows = []
    for part, selected in frame.groupby("part", sort=False):
        group = selected.loc[selected.eligible]
        row = {
            "part": part,
            "selected_rows": len(selected),
            "eligible_rows": len(group),
            "excluded_rows": len(selected) - len(group),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"median_{metric}"] = float(values.median()) if len(values) else float("nan")
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else float("nan")
        row["fraction_context_evidence_without_part_positive"] = (
            float((group.context_evidence_without_part > 0).mean())
            if len(group) else float("nan"))
        row["fraction_part_response_with_context_positive"] = (
            float((group.part_response_with_context > 0).mean())
            if len(group) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
