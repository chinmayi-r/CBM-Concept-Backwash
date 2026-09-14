#!/usr/bin/env python3
"""Small all-to-all audit of accepted FunnyBird Standard-CBM swaps.

The accepted CSV contains the original/counterfactual RGB paths and the two
scores directly involved in each swap. This script replays a balanced subset
through the unchanged accepted Koh Joint checkpoint to recover all 26 raw
concept logits before and after. It then measures every changed-input-part to
output-concept-block pathway. No model is trained or fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
for path in (
    HERE,
    CURATED / "external" / "ConceptBottleneck",
    CURATED / "compat",
    CURATED / "data" / "funnybirds",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import funnybirds_concepts as fbc  # noqa: E402
from funnybird_allpart_swap_preflight_core import (  # noqa: E402
    balanced_swap_rows, decompose_swap, summarize_pathways,
)


DEFAULT_PARTS = ["tail", "wing", "beak", "foot", "eye"]
PART_COLORS = {
    "tail": "#6f0db7", "wing": "#0077b6", "beak": "#eea400",
    "foot": "#009e73", "eye": "#c774a5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funnybirds-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--swap-root", required=True)
    parser.add_argument("--csv-name", default="funnybirds-cbm-s1.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts", nargs="+", default=DEFAULT_PARTS)
    parser.add_argument("--rows-per-input-part", type=int, default=18)
    parser.add_argument("--strict-replay-tolerance", type=float, default=0.02)
    parser.add_argument("--maximum-replay-tolerance", type=float, default=0.05)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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


def load_koh_model(checkpoint: Path, device: torch.device):
    """Load only the accepted 26-logit Koh Joint ResNet-50 structure."""
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
            f"unexpected Koh class-head shape {head.in_features}->{head.out_features}; "
            "expected 26->50")
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


def make_run_fn(model, device: torch.device):
    # Import here so schema/unit checks and --help do not require the image stack.
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2.0, 2.0, 2.0]),
    ])

    @torch.inference_mode()
    def run(image: Image.Image) -> np.ndarray:
        output = model(transform(image).unsqueeze(0).to(device))
        if not isinstance(output, (list, tuple)) or len(output) != 27:
            raise RuntimeError(
                "unexpected Koh output contract; expected class output plus 26 concepts")
        logits = torch.cat(
            [value.reshape(value.shape[0], -1) for value in output[1:]], dim=1)
        if logits.shape != (1, 26):
            raise RuntimeError(f"checkpoint emitted concept shape {tuple(logits.shape)}")
        return logits[0].detach().float().cpu().numpy()

    return run


def outcome(margin: float, response_delta: float) -> str:
    if margin > 0:
        return "donor wins"
    if response_delta > 0:
        return "donorward, source wins"
    return "no donorward move"


def infer_unique(selected: pd.DataFrame, run) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    orig_records = selected[["image_orig_sha256", "image_orig_path"]].drop_duplicates()
    cf_records = selected[["image_cf_sha256", "image_cf_path"]].drop_duplicates()
    if orig_records.groupby("image_orig_sha256").image_orig_path.nunique().max() != 1:
        raise RuntimeError("one original RGB hash maps to multiple paths")
    if cf_records.groupby("image_cf_sha256").image_cf_path.nunique().max() != 1:
        raise RuntimeError("one counterfactual RGB hash maps to multiple paths")

    def infer(records: pd.DataFrame, sha_column: str, path_column: str,
              label: str) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        for number, row in enumerate(records.itertuples(index=False), 1):
            sha = str(getattr(row, sha_column))
            path = require_file(Path(getattr(row, path_column)))
            actual = file_sha256(path)
            if actual != sha:
                raise RuntimeError(
                    f"accepted {label} RGB bytes changed: {path}; "
                    f"expected={sha} actual={actual}")
            with Image.open(path) as image:
                logits = run(image.convert("RGB"))
            if logits.shape != (26,) or not np.isfinite(logits).all():
                raise RuntimeError(f"{label} replay emitted invalid shape/values for {path}")
            values[sha] = logits
            if number % 25 == 0 or number == len(records):
                print(f"{label} frozen replay {number}/{len(records)}", flush=True)
        return values

    return (
        infer(orig_records, "image_orig_sha256", "image_orig_path", "original"),
        infer(cf_records, "image_cf_sha256", "image_cf_path", "counterfactual"),
    )


def matrix(summary: pd.DataFrame, value: str, parts: list[str]) -> np.ndarray:
    return (summary.pivot(index="input_part", columns="output_part", values=value)
            .reindex(index=parts, columns=parts).to_numpy(float))


def heatmap(ax, values: np.ndarray, parts: list[str], title: str,
            *, diverging: bool, blank_diagonal: bool = False,
            fmt: str = ".2f") -> None:
    shown = values.copy()
    if blank_diagonal:
        np.fill_diagonal(shown, np.nan)
    if diverging:
        limit = float(np.nanmax(np.abs(shown))) if np.isfinite(shown).any() else 1.0
        limit = max(limit, 1e-9)
        image = ax.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(shown, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(parts)), parts)
    ax.set_yticks(range(len(parts)), parts)
    ax.set_xlabel("output concept block whose scores are measured")
    ax.set_ylabel("part physically replaced in the image")
    ax.set_title(title)
    for row in range(len(parts)):
        for col in range(len(parts)):
            value = shown[row, col]
            if np.isfinite(value):
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(col, row, format(value, fmt), ha="center", va="center",
                        fontsize=8, color="black" if luminance > 0.55 else "white")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_figure(summary: pd.DataFrame, parts: list[str], out: Path) -> None:
    movement = matrix(summary, "mean_absolute_score_change", parts)
    class_use = matrix(summary, "mean_class_gap_shift", parts)
    unchanged = matrix(summary, "mean_unchanged_margin_change", parts)
    diagonal = (summary.loc[summary.input_part == summary.output_part]
                .set_index("input_part").reindex(parts))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    heatmap(
        axes[0, 0], movement, parts,
        "A · Mean |raw-score change| per coordinate", diverging=False)
    heatmap(
        axes[0, 1], unchanged, parts,
        "B · Change in correctness margin of unchanged blocks",
        diverging=True, blank_diagonal=True)
    heatmap(
        axes[1, 0], class_use, parts,
        "C · Saved-head source-minus-donor gap shift", diverging=True)
    axes[1, 1].bar(
        parts, diagonal.mean_target_response_delta.to_numpy(float),
        color=[PART_COLORS[part] for part in parts])
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("D · Intended donor-vs-source response of changed block")
    axes[1, 1].set_ylabel("raw-logit response; positive follows inserted part")
    axes[1, 1].set_xlabel("part physically replaced in the image")
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "FunnyBird Standard CBM · small all-part controlled-swap pathway preflight",
        fontsize=15)
    fig.savefig(out / "figure_1_allpart_swap_pathways.png", dpi=180)
    plt.close(fig)


def write_method(out: Path, args: argparse.Namespace, coverage: pd.DataFrame) -> None:
    text = f"""# FunnyBird all-part controlled-swap pathway preflight

This run reuses the accepted seed-1 Koh Joint ResNet-50 Standard CBM and its
accepted renderer swaps. It trains no model and fits no diagnostic classifier.
It replays {args.rows_per_input_part} counterfactual rows for each of the five
input parts, balanced round-robin across that part's inserted exact values.

The point is not another tail-only score. The point is to inspect all 25 paths:
each physically changed input part (`tail`, `wing`, `beak`, `foot`, `eye`) into
each output concept block with those same five names.

## Quantities

For a swap row, `z_orig` is the frozen model's 26 raw concept scores on the
original image and `z_cf` is its 26 scores after one part is replaced.

1. **Raw movement in output block g** is
   `mean_j in g |z_cf[j] - z_orig[j]|`. Dividing by block width makes a
   three-coordinate eye block and a nine-coordinate tail block comparable.
2. **Unchanged-block correctness margin** is
   `z[original value] - max(z[every other value in that block])`.
   Panel B plots counterfactual minus original margin. It is blank on the
   diagonal because the changed block should adopt the donor value. Negative
   off-diagonal values mean changing another part damaged an unchanged part.
3. The saved linear species head is `class_logits = W z + b`. For each output
   block g, its contribution to the change in the source-minus-donor species
   gap is `(W[source,g] - W[donor,g]) dot (z_cf[g] - z_orig[g])`. Positive
   means that block pushes the saved head toward the old source species;
   negative means it pushes toward the donor species. The five block
   contributions are checked to sum exactly to the full saved-head gap change.
4. On the diagonal only, intended swap response is
   `[(z_donor-z_source)_cf - (z_donor-z_source)_orig]`. Positive means the
   changed concept block followed the inserted pixels.

Concrete example: if an unchanged wing's original-value score falls from 4 to
2 while its strongest wrong wing rises from 1 to 2.5, its correctness margin
changes from `4-1=3` to `2-2.5=-0.5`, so Panel B records `-0.5-3=-3.5` for that
row. That is cross-part interference: a non-wing edit made the wing block wrong.

The heatmap row is the physical input change. The column is the output block
being measured. Every printed cell is an average over the rows listed in
`pathway_rows.csv`; no image row is treated as an uncertainty replicate.

## Sample coverage

```text
{coverage.to_string(index=False)}
```

This is a mechanism preflight, not a final causal decomposition. It can identify
which all-part pathways deserve full 5,000-row analysis or a correction test.
It cannot by itself prove why training created a pathway.
"""
    (out / "METHOD.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    funnybirds = Path(args.funnybirds_root).resolve()
    checkpoint = require_file(Path(args.checkpoint).resolve())
    swap_csv = require_file(Path(args.swap_root).resolve() / args.csv_name)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to mix with non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    parts_json = require_file(funnybirds / "parts.json")
    annotations_path = require_file(funnybirds / "dataset_test.json")
    parts_spec = fbc.load_parts(funnybirds)
    unknown = sorted(set(args.parts) - set(parts_spec))
    if unknown:
        raise ValueError(f"unknown parts {unknown}; available={list(parts_spec)}")
    if len(set(args.parts)) != len(args.parts):
        raise ValueError("--parts contains duplicates")
    spans = fbc.group_slices(parts_spec)
    lookup = fbc.build_part_lookup(parts_spec)
    parts_with_color = {
        part for part, variants in parts_spec.items()
        if any("color" in value for value in variants)
    }
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))

    swaps = pd.read_csv(swap_csv)
    required = {
        "part", "var_src", "var_donor", "sid_src", "sid_donor", "li",
        "render_id", "orig_render_id", "image_orig_path", "image_orig_sha256",
        "image_cf_path", "image_cf_sha256", "z_old_orig", "z_new_orig",
        "z_old", "z_new", "margin", "response_delta",
    }
    missing = required - set(swaps)
    if missing:
        raise RuntimeError(
            f"accepted swap CSV lacks required columns {sorted(missing)}; "
            f"available={list(swaps.columns)}")
    counts = swaps.groupby("part").size().reindex(args.parts)
    if counts.isna().any() or (counts <= 0).any():
        raise RuntimeError(f"swap population missing requested parts: {counts.to_dict()}")
    if len(swaps) != 5000 or swaps.orig_render_id.nunique() != 250:
        raise RuntimeError(
            f"expected accepted 5,000-row/250-original population, got "
            f"{len(swaps)} rows/{swaps.orig_render_id.nunique()} originals")

    selected = balanced_swap_rows(swaps, args.parts, args.rows_per_input_part)
    coverage = (selected.groupby("part", sort=False)
                .agg(rows=("render_id", "size"),
                     originals=("orig_render_id", "nunique"),
                     source_values=("var_src", "nunique"),
                     donor_values=("var_donor", "nunique"),
                     source_species=("sid_src", "nunique"),
                     donor_species=("sid_donor", "nunique"))
                .reset_index())
    expected_widths = {part: spans[part][1] - spans[part][0] for part in args.parts}
    for row in coverage.itertuples(index=False):
        if row.donor_values != expected_widths[row.part]:
            raise RuntimeError(
                f"{row.part}: selected {row.donor_values} donor values; "
                f"expected all {expected_widths[row.part]}")
    print("REAL INPUT COVERAGE:")
    print(coverage.to_string(index=False))
    selected.to_csv(out / "selected_swap_rows.csv", index=False)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "accepted-swap pathway replay requires CUDA for matched frozen inference; "
            "no training is performed")
    device = torch.device("cuda")
    model = load_koh_model(checkpoint, device)
    run = make_run_fn(model, device)
    class_weights = model.sec_model.linear.weight.detach().float().cpu().numpy()
    original_z, cf_z = infer_unique(selected, run)

    pathway_rows: list[dict] = []
    audit_rows: list[dict] = []
    for row in selected.itertuples(index=False):
        original = original_z[str(row.image_orig_sha256)]
        counterfactual = cf_z[str(row.image_cf_sha256)]
        input_lo, _ = spans[row.part]
        source_global = input_lo + int(row.var_src)
        donor_global = input_lo + int(row.var_donor)
        checks = {
            "original source": (float(original[source_global]), float(row.z_old_orig)),
            "original donor": (float(original[donor_global]), float(row.z_new_orig)),
            "counterfactual source": (float(counterfactual[source_global]), float(row.z_old)),
            "counterfactual donor": (float(counterfactual[donor_global]), float(row.z_new)),
        }
        errors = {name: abs(current - accepted)
                  for name, (current, accepted) in checks.items()}
        replay_m_orig = float(original[donor_global] - original[source_global])
        replay_m_cf = float(counterfactual[donor_global] - counterfactual[source_global])
        replay_response = replay_m_cf - replay_m_orig
        audit_rows.append({
            "render_id": row.render_id, "part": row.part,
            **{f"{name}_absolute_error": value for name, value in errors.items()},
            "maximum_absolute_error": max(errors.values()),
            "accepted_outcome": outcome(float(row.margin), float(row.response_delta)),
            "replayed_outcome": outcome(replay_m_cf, replay_response),
            "outcome_agrees": (
                outcome(float(row.margin), float(row.response_delta)) ==
                outcome(replay_m_cf, replay_response)),
        })

        image_index = int(row.li)
        if not 0 <= image_index < len(annotations):
            raise RuntimeError(f"swap li={image_index} outside dataset_test.json")
        ann = annotations[image_index]
        if int(ann["class_idx"]) != int(row.sid_src):
            raise RuntimeError(
                f"{row.render_id}: annotation species {ann['class_idx']} != source {row.sid_src}")
        original_values = {
            part: variant_index(ann, part, lookup, parts_with_color)
            for part in spans
        }
        if any(value < 0 for value in original_values.values()):
            raise RuntimeError(
                f"{row.render_id}: original annotation lacks a named-part value: "
                f"{original_values}")
        if original_values[row.part] != int(row.var_src):
            raise RuntimeError(
                f"{row.render_id}: annotation {row.part} value "
                f"{original_values[row.part]} != CSV source value {row.var_src}")

        decomposed = decompose_swap(
            z_orig=original, z_cf=counterfactual, class_weights=class_weights,
            source_species=int(row.sid_src), donor_species=int(row.sid_donor),
            input_part=str(row.part), source_value=int(row.var_src),
            donor_value=int(row.var_donor), original_values=original_values,
            spans=spans)
        for item in decomposed:
            item.update({
                "render_id": row.render_id,
                "original_image": row.orig_render_id,
                "source_species": int(row.sid_src),
                "donor_species": int(row.sid_donor),
                "source_value": int(row.var_src),
                "donor_value": int(row.var_donor),
            })
            pathway_rows.append(item)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "replay_audit.csv", index=False)
    maximum_error = float(audit.maximum_absolute_error.max())
    strict_rows = int((audit.maximum_absolute_error > args.strict_replay_tolerance).sum())
    disagreement_rows = int((~audit.outcome_agrees).sum())
    print(
        "REPLAY AUDIT:",
        {"rows": len(audit), "max_absolute_score_error": maximum_error,
         "rows_above_strict_tolerance": strict_rows,
         "outcome_disagreements": disagreement_rows})
    if maximum_error > args.maximum_replay_tolerance:
        raise RuntimeError(
            f"maximum replay error {maximum_error:.6g} exceeds disclosed cap "
            f"{args.maximum_replay_tolerance}")

    pathways = pd.DataFrame(pathway_rows)
    summary = summarize_pathways(pathways)
    pathways.to_csv(out / "pathway_rows.csv", index=False)
    summary.to_csv(out / "pathway_summary.csv", index=False)
    cross = summary.loc[summary.input_part != summary.output_part].copy()
    cross["absolute_mean_class_gap_shift"] = cross.mean_class_gap_shift.abs()
    cross["absolute_mean_unchanged_margin_change"] = (
        cross.mean_unchanged_margin_change.abs())
    cross = cross.sort_values(
        ["absolute_mean_class_gap_shift", "mean_absolute_score_change"],
        ascending=False)
    cross.to_csv(out / "ranked_cross_part_cells.csv", index=False)
    save_figure(summary, args.parts, out)
    write_method(out, args, coverage)

    status = {
        "status": "ACCEPTED FOR SMALL ALL-PART MECHANISM PREFLIGHT",
        "scientific_result": False,
        "visual_review_required": True,
        "training": False,
        "diagnostic_fit": False,
        "parts": args.parts,
        "selected_swap_rows": len(selected),
        "pathway_rows": len(pathways),
        "max_absolute_score_replay_error": maximum_error,
        "rows_above_strict_replay_tolerance": strict_rows,
        "outcome_disagreements": disagreement_rows,
        "checkpoint_sha256": file_sha256(checkpoint),
        "swap_csv_sha256": file_sha256(swap_csv),
        "parts_json_sha256": file_sha256(parts_json),
    }
    (out / "PREFLIGHT_STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ALL-PART PATHWAY PREFLIGHT COMPLETE:", out)
    print("Top cross-part cells by saved-head gap shift magnitude:")
    print(cross.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
