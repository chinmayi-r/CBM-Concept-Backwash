"""Silhouette-preserving mask replacement pilot; no model or training imports.

This is a new, uncalibrated editing method, not a claim of causal validity.
The source-only hole is filled from nearest context pixels. That approximation
is explicitly measured and must survive self/same-value and renderer controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, map_coordinates


def image_hash(rgb):
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def validate(rgb, mask):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB must be uint8 [height,width,3]")
    if mask.dtype != bool or mask.shape != rgb.shape[:2] or not mask.any():
        raise ValueError("mask must be nonempty bool on the exact RGB grid")
    if mask.all():
        raise ValueError("whole-image masks cannot support context-preserving replacement")


def similarity_matrix(source_mask, donor_mask, source_anchors=None, donor_anchors=None):
    """Map donor to source; isotropic scale preserves donor shape proportions.

    Two ordered anchors define an attachment and a direction. Without anchors,
    align mask centroids and equalize area without rotating or reflecting.
    The latter is a declared candidate transform, not verified pose alignment.
    """
    if (source_anchors is None) != (donor_anchors is None):
        raise ValueError("provide both anchor pairs or neither")
    if source_anchors is None:
        s = np.argwhere(source_mask)[:, ::-1].mean(0)
        d = np.argwhere(donor_mask)[:, ::-1].mean(0)
        a = np.eye(2) * np.sqrt(source_mask.sum() / donor_mask.sum())
    else:
        sp, dp = np.asarray(source_anchors, float), np.asarray(donor_anchors, float)
        if sp.shape != (2, 2) or dp.shape != (2, 2) or not np.isfinite([sp, dp]).all():
            raise ValueError("ordered anchors must be two finite x,y points")
        sv, dv = sp[1] - sp[0], dp[1] - dp[0]
        if min(np.linalg.norm(sv), np.linalg.norm(dv)) < 1:
            raise ValueError("anchor baseline is too short")
        angle = np.arctan2(sv[1], sv[0]) - np.arctan2(dv[1], dv[0])
        a = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]) * (np.linalg.norm(sv) / np.linalg.norm(dv))
        s, d = sp[0], dp[0]
    return np.column_stack([a, s - a @ d])


def replace_part(source, source_mask, donor, donor_mask, source_anchors=None, donor_anchors=None):
    validate(source, source_mask); validate(donor, donor_mask)
    matrix = similarity_matrix(source_mask, donor_mask, source_anchors, donor_anchors)
    yy, xx = np.indices(source_mask.shape)
    donor_xy = np.linalg.solve(matrix[:, :2], np.stack([xx.ravel(), yy.ravel()]) - matrix[:, 2, None])
    coords = [donor_xy[1].reshape(xx.shape), donor_xy[0].reshape(xx.shape)]
    placed_mask = map_coordinates(donor_mask.astype(float), coords, order=0,
                                  mode="constant", cval=0, prefilter=False) > .5
    if not placed_mask.any():
        raise ValueError("transformed donor mask lies outside source image")
    # Nearest donor-part pixels prevent interpolation with donor background.
    _, nearest = distance_transform_edt(~donor_mask, return_indices=True)
    donor_texture = donor[nearest[0], nearest[1]]
    placed = np.stack([map_coordinates(donor_texture[..., c].astype(float), coords,
                      order=1, mode="nearest", prefilter=False) for c in range(3)], -1)
    allowed = source_mask | placed_mask
    if allowed.all():
        raise ValueError("replacement leaves no unchanged context")
    hole = source_mask & ~placed_mask
    result = source.copy()
    # Fill from outside both old and inserted part; never retain old-part texture.
    _, context = distance_transform_edt(allowed, return_indices=True)
    result[hole] = source[context[0][hole], context[1][hole]]
    result[placed_mask] = np.clip(np.rint(placed[placed_mask]), 0, 255).astype(np.uint8)
    changed = np.any(result != source, axis=-1)
    if np.any(changed & ~allowed):
        raise RuntimeError("unexpected off-target pixel change")
    expected_area = float(donor_mask.sum() * np.linalg.det(matrix[:, :2]))
    metrics = dict(transform=matrix.tolist(), alignment="anchors" if source_anchors is not None else "centroid_equal_area",
                   source_pixels=int(source_mask.sum()), inserted_pixels=int(placed_mask.sum()),
                   expected_inserted_area=expected_area, raster_area_ratio=float(placed_mask.sum()/expected_area),
                   source_only_filled_pixels=int(hole.sum()), changed_pixels=int(changed.sum()),
                   outside_allowed_changed_pixels=0, original_rgb_sha256=image_hash(source),
                   edited_rgb_sha256=image_hash(result), fill_method="nearest_unchanged_context",
                   status="METHOD NOT CALIBRATED")
    return result, placed_mask, allowed, metrics


def load_pair(image_path, mask_path):
    return np.asarray(Image.open(image_path).convert("RGB")), np.asarray(Image.open(mask_path).convert("L")) > 0


def generate_trial(trial, output):
    source, sm = load_pair(trial["source_image"], trial["source_mask"])
    output.mkdir(parents=True, exist_ok=False)
    panels, stats = [], {}
    Image.fromarray(source).save(output / "original.png")
    panels.append(("Original", source))
    for variant in ("self", "same_value", "different_value"):
        donor, dm = (source, sm) if variant == "self" else load_pair(trial[f"{variant}_image"], trial[f"{variant}_mask"])
        sa = trial.get("source_anchors")
        da = sa if variant == "self" else trial.get(f"{variant}_anchors")
        edited, placed, allowed, metrics = replace_part(source, sm, donor, dm, sa, da)
        Image.fromarray(edited).save(output / f"{variant}.png")
        Image.fromarray(placed.astype(np.uint8)*255).save(output / f"{variant}_inserted_mask.png")
        Image.fromarray(allowed.astype(np.uint8)*255).save(output / f"{variant}_allowed_mask.png")
        stats[variant] = metrics
        panels.append((variant, edited))
    if stats["self"]["changed_pixels"]:
        raise RuntimeError("self reinsertion is not pixel-identical")
    width, height = source.shape[1], source.shape[0]
    sheet = Image.new("RGB", (width*4, height+56), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 3), f"Example {trial['trial_id']} | METHOD NOT CALIBRATED | no model-based selection", fill="black")
    for i,(label,rgb) in enumerate(panels):
        draw.text((i*width+5, 27), label, fill="black")
        sheet.paste(Image.fromarray(rgb), (i*width,56))
    sheet.save(output / "review.png")
    manifest = dict(trial=trial, edits=stats, training=False, model_scores_used_for_selection=False,
                    status="METHOD NOT CALIBRATED", review_required="inserted value, pose, holes, co-changing attributes, crop visibility")
    (output / "EDIT.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    trials=json.loads(args.plan.read_text())
    ids=[t["trial_id"] for t in trials]
    if not ids or len(set(ids))!=len(ids) or any(not str(i).replace("-", "").replace("_", "").isalnum() for i in ids):
        raise ValueError("trial IDs must be unique safe names; plan must be nonempty")
    args.out.mkdir(parents=True,exist_ok=False)
    for i,t in enumerate(trials):
        print(f"Editing trial {i+1}/{len(trials)}: {t['trial_id']}; no training or model-score selection",flush=True)
        generate_trial(t,args.out/str(t["trial_id"]))


if __name__=="__main__":main()
