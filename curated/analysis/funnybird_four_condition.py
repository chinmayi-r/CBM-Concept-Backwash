#!/usr/bin/env python3
"""FunnyBird four-condition grounding diagnostic for an accepted Koh Joint CBM.

For each selected ordinary image and named part, evaluate a symmetric four-cell
pixel factorial:

    z11: identical target pixels pasted onto a base with the other named parts
    z01: the same base without the target pixels
    z10: identical target pixels pasted onto a base without other named parts
    z00: the same base without the target pixels

The target pixels in z11 and z10 are byte-identical and both present cells use
the same paste operator. Camera, lighting, pose, base body, and background stay
fixed. "Other named parts absent" means only the four other FunnyBird part meshes
are removed; it is not a generic removal of all species/body context. Native
renderer outputs are retained only to calibrate the composites. The script reuses
a frozen checkpoint and never trains. Every selected row remains in rows.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from base64 import decodebytes
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
for path in (CURATED / "external" / "ConceptBottleneck",
             CURATED / "compat",
             CURATED / "data" / "funnybirds"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import funnybirds_concepts as fbc  # noqa: E402
from funnybird_four_condition_core import (  # noqa: E402
    add_contrasts, exact_target_composite, four_conditions,
    image_rgb_mae, summarize,
)


PART_SEG_COLORS = {
    "beak": ((255, 255, 0),),
    "eye": ((255, 255, 253), (255, 255, 254)),
    "wing": ((0, 255, 1), (0, 255, 2)),
    "foot": ((255, 0, 1), (255, 0, 2)),
    "tail": ((0, 0, 255),),
}
CONDITION_ORDER = ("11", "01", "10", "00")
CONDITION_TITLES = {
    "11": "11 · target pasted\nother named parts present",
    "01": "01 · target absent\nother named parts present",
    "10": "10 · target pasted\nother named parts absent",
    "00": "00 · target absent\nother named parts absent",
}
PART_COLORS = {
    "tail": "#6f0db7", "wing": "#0077b6", "beak": "#eea400",
    "foot": "#009e73", "eye": "#c774a5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funnybirds-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--renderer-url", default="http://localhost:8081")
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts", nargs="+", default=["tail", "wing"])
    parser.add_argument("--max-images-per-part", type=int, default=25,
                        help="0 means all eligible test images")
    parser.add_argument("--gallery-images-per-part", type=int, default=4)
    parser.add_argument("--min-target-pixels", type=int, default=8)
    parser.add_argument("--min-context-changed-pixels", type=int, default=8)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--min-valid-per-variant", type=int, default=1)
    parser.add_argument("--request-retries", type=int, default=4)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ann_digest(ann: dict) -> str:
    payload = json.dumps(ann, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)[:16]


def json_to_url(prefix: str, sample: dict, render_mode: str = "default") -> str:
    # Keep the accepted renderer's field serialization exactly. requests handles
    # percent escaping for the final request, while the server accepts empty fields.
    fields = [f"render_mode={render_mode}"]
    fields.extend(f"{key}={value}" for key, value in sample.items()
                  if key != "class_idx")
    return prefix.rstrip("/") + "/render?" + "&".join(fields)


def render(prefix: str, ann: dict, part_map: bool, retries: int) -> Image.Image:
    mode = "part_map" if part_map else "default"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(json_to_url(prefix, ann, mode), timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(decodebytes(response.content))).convert("RGB")
            return image.resize((256, 256), Image.NEAREST if part_map else Image.BILINEAR)
        except Exception as exc:  # preserve the final concrete renderer error
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"renderer failed after {retries} attempts: {last_error}")


def target_mask(part_map: Image.Image, part: str) -> np.ndarray:
    array = np.asarray(part_map.convert("RGB"))
    mask = np.zeros(array.shape[:2], dtype=bool)
    for color in PART_SEG_COLORS[part]:
        mask |= np.all(array == np.asarray(color, dtype=np.uint8), axis=2)
    return mask


def changed_pixels(a: Image.Image, b: Image.Image) -> int:
    aa = np.asarray(a.convert("RGB"), dtype=np.int16)
    bb = np.asarray(b.convert("RGB"), dtype=np.int16)
    return int((np.max(np.abs(aa - bb), axis=2) > 2).sum())


def variant_index(ann: dict, part: str, lookup: dict,
                  parts_with_color: set[str]) -> int:
    model = ann.get(f"{part}_model", "")
    if not model or model == "placeholder":
        return -1
    fields = {"model": model}
    if part in parts_with_color:
        color = ann.get(f"{part}_color", "")
        if color:
            fields["color"] = color
    return int(lookup[part][tuple(sorted(fields.items()))])


def select_balanced_indices(annotations: list[dict], part: str, lookup: dict,
                            parts_with_color: set[str], n_variants: int,
                            limit: int) -> list[int]:
    by_variant: dict[int, list[int]] = defaultdict(list)
    for index, ann in enumerate(annotations):
        variant = variant_index(ann, part, lookup, parts_with_color)
        if variant >= 0:
            by_variant[variant].append(index)
    if set(by_variant) != set(range(n_variants)):
        missing = sorted(set(range(n_variants)) - set(by_variant))
        raise RuntimeError(f"{part}: test annotations have no rows for variants {missing}")
    ordered: list[int] = []
    depth = 0
    target = len(annotations) if limit == 0 else limit
    while len(ordered) < target:
        added = False
        for variant in range(n_variants):
            values = by_variant[variant]
            if depth < len(values):
                ordered.append(values[depth])
                added = True
                if len(ordered) == target:
                    break
        if not added:
            break
        depth += 1
    return ordered


def load_koh_model(checkpoint: Path, device: torch.device):
    try:
        model = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint, map_location=device)
    if getattr(model, "curated_framework", None) != "koh_joint":
        raise RuntimeError("checkpoint is not marked as the accepted Koh Joint framework")
    if getattr(model, "curated_backbone", None) != "resnet50":
        raise RuntimeError("checkpoint is not marked as the accepted ResNet-50 backbone")
    if not hasattr(model, "first_model") or not hasattr(model, "sec_model"):
        raise RuntimeError("checkpoint lacks Koh first_model/sec_model structure")
    if not hasattr(model.sec_model, "linear"):
        raise RuntimeError("checkpoint has no Koh sec_model.linear class head")
    head = model.sec_model.linear
    if (head.in_features, head.out_features) != (26, 50):
        raise RuntimeError(
            "unexpected Koh class-head shape: "
            f"{head.in_features}->{head.out_features}; expected 26->50")
    if getattr(model, "use_relu", True) or getattr(model, "use_sigmoid", True):
        raise RuntimeError("class head is not using Koh's required raw concept logits")
    module_names = [
        f"{type(module).__module__}.{type(module).__name__}".lower()
        for module in model.modules()
    ]
    if any("inception" in name for name in module_names):
        raise RuntimeError("checkpoint unexpectedly contains an Inception module")
    if any("minimal_cbm" in name or ".mcbm" in name for name in module_names):
        raise RuntimeError("checkpoint unexpectedly contains an MCBM module")
    print("[MODEL STRUCTURE PASS] Koh Joint, ResNet-50, raw 26->50 linear head")
    return model.to(device).eval()


def make_run_fn(model, n_concepts: int, device: torch.device):
    transform = transforms.Compose([
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2.0, 2.0, 2.0]),
    ])

    @torch.inference_mode()
    def run(image: Image.Image) -> np.ndarray:
        output = model(transform(image).unsqueeze(0).to(device))
        concept_outputs = output[1:]
        logits = torch.cat(
            [value.reshape(value.shape[0], -1) for value in concept_outputs], dim=1)
        if logits.shape != (1, n_concepts):
            raise RuntimeError(
                f"checkpoint emitted concept shape {tuple(logits.shape)}; "
                f"expected (1, {n_concepts})")
        return logits[0].detach().float().cpu().numpy()

    return run


def renderer_alive(prefix: str) -> bool:
    probe = {
        "beak_model": "beak01.glb", "eye_model": "eye01.glb",
        "foot_model": "foot01.glb", "tail_model": "tail01.glb",
        "tail_color": "red", "wing_model": "wing01.glb", "wing_color": "red",
        "camera_distance": 300, "camera_pitch": 0, "camera_roll": 0,
        "light_distance": 300, "light_pitch": 0, "light_roll": 0,
    }
    try:
        return requests.get(json_to_url(prefix, probe), timeout=3).status_code == 200
    except Exception:
        return False


def save_gallery(records: list[dict], out: Path, per_part: int) -> None:
    chosen = []
    for part in dict.fromkeys(record["part"] for record in records):
        candidates = [r for r in records if r["part"] == part and r["eligible"]]
        chosen.extend(candidates[:per_part])
    if not chosen:
        chosen = records[:max(1, per_part)]
    cell_w, cell_h, label_h = 256, 256, 42
    sheet = Image.new("RGB", (4 * cell_w, len(chosen) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(chosen):
        y = row * (cell_h + label_h)
        for col, condition in enumerate(CONDITION_ORDER):
            x = col * cell_w
            draw.text((x + 4, y + 3),
                      f"{record['part']} · image {record['image_index']} · "
                      f"{CONDITION_TITLES[condition]}", fill="black")
            sheet.paste(record["images"][condition], (x, y + label_h))
    sheet.save(out / "figure_1_four_condition_gallery.png")


def save_calibration_gallery(records: list[dict], out: Path, per_part: int) -> None:
    """Show whether the paste reconstruction resembles each native render."""
    chosen = []
    for part in dict.fromkeys(record["part"] for record in records):
        candidates = [r for r in records if r["part"] == part]
        chosen.extend(candidates[:per_part])
    columns = (
        ("native_11", "native target + other parts"),
        ("11", "pasted target + other parts"),
        ("native_10", "native target, other parts absent"),
        ("10", "pasted target, other parts absent"),
    )
    cell_w, cell_h, label_h = 256, 256, 42
    sheet = Image.new(
        "RGB", (len(columns) * cell_w, len(chosen) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(chosen):
        y = row * (cell_h + label_h)
        for col, (key, title) in enumerate(columns):
            x = col * cell_w
            draw.text(
                (x + 4, y + 3),
                f"{record['part']} · image {record['image_index']}\n{title}",
                fill="black",
            )
            sheet.paste(record["images"][key], (x, y + label_h))
    sheet.save(out / "audit_A1_native_vs_composite.png")


def save_score_figure(frame: pd.DataFrame, out: Path) -> None:
    eligible = frame.loc[frame.eligible].copy()
    if eligible.empty:
        return
    parts = list(dict.fromkeys(eligible.part.tolist()))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    def grouped_box(ax, columns, labels, title, zero=True):
        width = 0.17
        centers = np.arange(len(parts), dtype=float)
        offsets = ([0.0] if len(columns) == 1
                   else np.linspace(-0.27, 0.27, len(columns)))
        for offset, column, label in zip(offsets, columns, labels):
            values = [eligible.loc[eligible.part == part, column].to_numpy() for part in parts]
            bp = ax.boxplot(values, positions=centers + offset, widths=width,
                            showfliers=False, patch_artist=True, manage_ticks=False)
            for patch, part in zip(bp["boxes"], parts):
                patch.set_facecolor(PART_COLORS.get(part, "#777777"))
                patch.set_alpha(0.32 + 0.15 * list(columns).index(column))
            bp["medians"][0].set_label(label)
        if zero:
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(centers, parts)
        ax.set_title(title)
        ax.set_ylabel("raw concept-logit units")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    grouped_box(
        axes[0, 0], ["z11", "z01", "z10", "z00"],
        ["z11", "z01", "z10", "z00"],
        "A · The four raw scores", zero=True)
    grouped_box(
        axes[0, 1], ["part_response_with_context", "part_response_without_context"],
        ["z11 − z01: part response with context",
         "z10 − z00: part response without context"],
        "B · What the named part contributes", zero=True)
    grouped_box(
        axes[1, 0], ["context_evidence_with_part", "context_evidence_without_part"],
        ["z11 − z10: context with part",
         "z01 − z00: context without part"],
        "C · What the four other named parts contribute", zero=True)
    grouped_box(
        axes[1, 1], ["interaction"],
        ["(z11 − z01) − (z10 − z00)"],
        "D · Do the other named parts change target response?", zero=True)
    fig.suptitle("FunnyBird Standard CBM · four-condition visual-calibration pilot", fontsize=15)
    fig.savefig(out / "figure_2_four_condition_scores.png", dpi=180)
    plt.close(fig)


def write_method(out: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = [
        "# FunnyBird four-condition pilot",
        "",
        "This is frozen inference on the accepted Koh Joint Standard CBM. It performs no training.",
        "",
        "Here `context` has one narrow meaning: the four other named FunnyBird part meshes. "
        "Removing them does not remove the base body, pose, camera, lighting, or background. "
        "This pilot therefore tests dependence on the other named parts conditional on all of "
        "that retained information; it does not claim to erase every species/body cue.",
        "",
        "The two target-absent native renders are `01` (other parts present) and `00` (other "
        "parts absent). We take the visible target pixels and mask from the native `11` render, "
        "then apply the same paste operation twice: onto `01` to make scientific `11`, and onto "
        "`00` to make scientific `10`. Thus `11` and `10` contain byte-identical target pixels. "
        "The native `11` and native `10` renders are saved only for calibration; neither enters "
        "the four-cell formulas.",
        "",
        "The five paired quantities are:",
        "",
        "- part response with context: `z11 - z01`;",
        "- part response without context: `z10 - z00`;",
        "- context evidence with the part: `z11 - z10`;",
        "- context evidence without the part: `z01 - z00`;",
        "- interaction: `(z11 - z01) - (z10 - z00)`.",
        "",
        "Example: if the four target scores are `z11=6`, `z01=4`, `z10=1`, and "
        "`z00=-3`, the target-pixel response is `6-4=2` with the other parts and "
        "`1-(-3)=4` without them. The other-part contribution is `6-1=5` when the "
        "target pixels are present and `4-(-3)=7` when they are absent. The interaction "
        "is `2-4=-2`: in this image, adding the other named parts makes the model two "
        "raw-logit units less responsive to the unchanged target pixels.",
        "",
        "A row is mechanically valid only if the native `11` target mask contains at least "
        f"{args.min_target_pixels} pixels, the target is absent from both base renders, both "
        "pastes visibly change their base, and removing the four other named parts changes at "
        f"least {args.min_context_changed_pixels} pixels. These checks prevent empty or unchanged "
        "renders from entering the summaries. They do not prove that the composites look natural.",
        "",
        "`audit_A1_native_vs_composite.png` and the native/composite difference columns in "
        "`rows.csv` are the calibration evidence. The run is only a pilot awaiting visual "
        "review. It is not accepted scientific evidence merely because the mechanical checks pass.",
        "",
        "## Counts",
        "",
        "```text",
        summary.to_string(index=False) if not summary.empty else "No selected rows.",
        "```",
        "",
        "No result interpretation is prewritten here. Review the current figures and tables first.",
        "",
    ]
    (out / "METHOD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    funnybirds = Path(args.funnybirds_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to mix with non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for required in (funnybirds / "parts.json", funnybirds / "dataset_test.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not renderer_alive(args.renderer_url):
        raise RuntimeError(f"renderer is not responding at {args.renderer_url}")

    parts = fbc.load_parts(funnybirds)
    unknown = sorted(set(args.parts) - set(parts))
    if unknown:
        raise ValueError(f"unknown FunnyBird parts: {unknown}; available={list(parts)}")
    lookup = fbc.build_part_lookup(parts)
    spans = fbc.group_slices(parts)
    concept_names = fbc.concept_names(parts)
    parts_with_color = {
        part for part, variants in parts.items() if any("color" in value for value in variants)}
    annotations = json.loads((funnybirds / "dataset_test.json").read_text())
    if not annotations:
        raise RuntimeError("dataset_test.json is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={checkpoint}")
    print(f"parts={args.parts} max_images_per_part={args.max_images_per_part}")
    model = load_koh_model(checkpoint, device)
    run = make_run_fn(model, len(concept_names), device)

    rows: list[dict] = []
    gallery_records: list[dict] = []
    image_root = out / "renders"
    image_root.mkdir()
    for part in args.parts:
        start, stop = spans[part]
        selected = select_balanced_indices(
            annotations, part, lookup, parts_with_color, stop - start,
            args.max_images_per_part)
        print(f"{part}: selected {len(selected)} annotations, balanced round-robin by exact value")
        for position, image_index in enumerate(selected, 1):
            ann = annotations[image_index]
            variant = variant_index(ann, part, lookup, parts_with_color)
            concept_index = start + variant
            conditions = four_conditions(ann, part, parts.keys(), parts_with_color)
            native_images, native_maps = {}, {}
            images, logits = {}, {}
            condition_shas = {}
            row_dir = image_root / part / f"{image_index:06d}"
            row_dir.mkdir(parents=True, exist_ok=True)
            # Render every native condition for calibration. The scientific
            # factorial below uses native 01/00 as target-absent bases and the
            # same exact-pixel insertion for both target-present cells.
            for condition in CONDITION_ORDER:
                image = render(args.renderer_url, conditions[condition], False,
                               args.request_retries)
                part_map = render(args.renderer_url, conditions[condition], True,
                                  args.request_retries)
                native_images[condition] = image
                native_maps[condition] = part_map
                image.save(row_dir / f"native_{condition}.png")
                part_map.save(row_dir / f"native_{condition}_partmap.png")

            native_masks = {
                condition: target_mask(native_maps[condition], part)
                for condition in CONDITION_ORDER
            }
            counts = {
                condition: int(mask.sum())
                for condition, mask in native_masks.items()
            }
            target = native_masks["11"]
            images["01"] = native_images["01"]
            images["00"] = native_images["00"]
            images["11"] = exact_target_composite(
                native_images["11"], native_images["01"], target)
            images["10"] = exact_target_composite(
                native_images["11"], native_images["00"], target)

            for condition in CONDITION_ORDER:
                logits[condition] = run(images[condition])
                image_bytes = io.BytesIO()
                images[condition].save(image_bytes, format="PNG")
                condition_shas[condition] = sha256_bytes(image_bytes.getvalue())
                images[condition].save(row_dir / f"{condition}.png")

            native_logits = {
                condition: run(native_images[condition]) for condition in ("11", "10")
            }
            target_equal = np.array_equal(
                np.asarray(images["11"])[target], np.asarray(images["10"])[target])
            outside = ~target
            paste_11_outside_equal = np.array_equal(
                np.asarray(images["11"])[outside], np.asarray(images["01"])[outside])
            paste_10_outside_equal = np.array_equal(
                np.asarray(images["10"])[outside], np.asarray(images["00"])[outside])
            changed_11_01 = changed_pixels(images["11"], images["01"])
            changed_10_00 = changed_pixels(images["10"], images["00"])
            changed_01_00 = changed_pixels(images["01"], images["00"])
            reasons = []
            if counts["11"] < args.min_target_pixels:
                reasons.append("target_not_visible_in_native_11")
            if counts["01"] != 0:
                reasons.append("target_still_present_in_native_01")
            if counts["00"] != 0:
                reasons.append("target_still_present_in_native_00")
            if not target_equal:
                reasons.append("scientific_target_pixels_not_identical")
            if not paste_11_outside_equal or not paste_10_outside_equal:
                reasons.append("paste_changed_pixels_outside_target_mask")
            if changed_11_01 < args.min_target_pixels:
                reasons.append("11_to_01_did_not_change_rgb")
            if changed_10_00 < args.min_target_pixels:
                reasons.append("10_to_00_did_not_change_rgb")
            if changed_01_00 < args.min_context_changed_pixels:
                reasons.append("other_named_parts_removal_did_not_change_rgb")

            row = {
                "part": part,
                "image_index": image_index,
                "species": int(ann["class_idx"]),
                "variant": variant,
                "concept_index": concept_index,
                "concept": concept_names[concept_index],
                "mechanically_valid": not reasons,
                "eligible": not reasons,
                "exclusion_reason": ";".join(reasons),
                "target_pixels_native_11": counts["11"],
                "target_pixels_native_01": counts["01"],
                "target_pixels_native_10": counts["10"],
                "target_pixels_native_00": counts["00"],
                "scientific_target_pixels_identical": target_equal,
                "changed_pixels_11_01": changed_11_01,
                "changed_pixels_10_00": changed_10_00,
                "changed_pixels_01_00_other_named_parts": changed_01_00,
                "composite_11_vs_native_11_changed_pixels": changed_pixels(
                    images["11"], native_images["11"]),
                "composite_10_vs_native_10_changed_pixels": changed_pixels(
                    images["10"], native_images["10"]),
                "composite_11_vs_native_11_rgb_mae": image_rgb_mae(
                    images["11"], native_images["11"]),
                "composite_10_vs_native_10_rgb_mae": image_rgb_mae(
                    images["10"], native_images["10"]),
                "native_z11": float(native_logits["11"][concept_index]),
                "native_z10": float(native_logits["10"][concept_index]),
                "composite_minus_native_z11": float(
                    logits["11"][concept_index] - native_logits["11"][concept_index]),
                "composite_minus_native_z10": float(
                    logits["10"][concept_index] - native_logits["10"][concept_index]),
                "annotation_sha256": ann_digest(ann),
            }
            for condition in CONDITION_ORDER:
                row[f"z{condition}"] = float(logits[condition][concept_index])
                row[f"render_sha256_{condition}"] = condition_shas[condition]
            rows.append(row)
            gallery_records.append({
                "part": part, "image_index": image_index,
                "eligible": not reasons,
                "images": {
                    **images,
                    "native_11": native_images["11"],
                    "native_10": native_images["10"],
                },
            })
            print(f"  {part} {position}/{len(selected)} image={image_index} "
                  f"variant={variant} eligible={not reasons}", flush=True)

    frame = add_contrasts(pd.DataFrame(rows))
    frame.to_csv(out / "rows.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out / "summary.csv", index=False)
    exclusions = (frame.groupby(["part", "exclusion_reason"], dropna=False)
                  .size().reset_index(name="rows"))
    exclusions.to_csv(out / "exclusions.csv", index=False)
    save_gallery(gallery_records, out, args.gallery_images_per_part)
    save_calibration_gallery(gallery_records, out, args.gallery_images_per_part)
    save_score_figure(frame, out)
    write_method(out, args, summary)

    mechanical_gate_failures = []
    for part, selected in frame.groupby("part", sort=False):
        valid = selected.loc[selected.eligible]
        valid_fraction = len(valid) / len(selected)
        if valid_fraction < args.min_valid_fraction:
            mechanical_gate_failures.append(
                f"{part}: valid fraction {valid_fraction:.3f} < {args.min_valid_fraction:.3f}")
        selected_variants = sorted(selected.variant.unique())
        valid_counts = valid.groupby("variant").size().to_dict()
        undercovered = [
            int(variant) for variant in selected_variants
            if int(valid_counts.get(variant, 0)) < args.min_valid_per_variant
        ]
        if undercovered:
            mechanical_gate_failures.append(
                f"{part}: fewer than {args.min_valid_per_variant} valid rows for variants "
                f"{undercovered}")

    status = (
        "METHOD NOT CALIBRATED"
        if mechanical_gate_failures
        else "ACCEPTED FOR VISUAL-CALIBRATION REVIEW ONLY"
    )
    pilot = {
        "status": status,
        "scientific_result": False,
        "scientific_scope": "four-condition frozen-inference visual-calibration pilot",
        "training": False,
        "model_framework": "Koh Joint CBM",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()),
        "funnybirds_root": str(funnybirds),
        "parts": args.parts,
        "selected_rows": int(len(frame)),
        "eligible_rows": int(frame.eligible.sum()),
        "excluded_rows": int((~frame.eligible).sum()),
        "mechanical_gate_failures": mechanical_gate_failures,
        "causal_boundary": (
            "other named-part meshes are removed, while base body, pose, camera, "
            "lighting, and background remain"
        ),
        "parameters": vars(args),
        "outputs": ["rows.csv", "summary.csv", "exclusions.csv",
                    "figure_1_four_condition_gallery.png",
                    "audit_A1_native_vs_composite.png",
                    "figure_2_four_condition_scores.png", "METHOD.md"],
    }
    (out / "PILOT_STATUS.json").write_text(json.dumps(pilot, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"[{status}] {out / 'PILOT_STATUS.json'}")
    if mechanical_gate_failures:
        for failure in mechanical_gate_failures:
            print(f"  {failure}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
