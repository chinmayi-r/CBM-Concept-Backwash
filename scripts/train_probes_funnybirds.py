#!/usr/bin/env python
"""
scripts/train_probes_funnybirds.py

Train layerwise linear probes on FunnyBirds features and save results to JSON.

Mirrors scripts/train_probes_attributes.py (CUB version) exactly in:
  - JSON output schema: layer, target_type, target_name, group,
                        train_acc, val_acc, best_val_acc
  - Probe architecture (LinearProbe via probe_utils)
  - Layer discovery from *_train.pt / *_test.pt files

Key differences from the CUB version:
  - Uses FunnyBirdsDataset metadata instead of CUB attribute tables
  - All 26 FunnyBirds concepts are used (no 10%-90% prevalence filter needed;
    the dataset is perfectly balanced by construction)
  - Concept groups = part names: beak, eye, wing, foot, tail
  - Species count = 50 (not 200)
  - Supports both baseline and CBM checkpoints via --features_dir

Usage (baseline):
    python -m scripts.train_probes_funnybirds \\
        --features_dir    features/resnet50_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --out_json        results/probes/resnet50_funnybirds_probes_fine.json

Usage (CBM backbone):
    python -m scripts.train_probes_funnybirds \\
        --features_dir    features/resnet50_cbm_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --out_json        results/probes/resnet50_funnybirds_cbm_probes_fine.json

Both outputs can then be loaded by the analysis notebooks
(02_ana_res.ipynb / 02_ana_jumpacc_res.ipynb style) as a drop-in for the
CUB probe JSONs, enabling direct comparison of emergence patterns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from datasets.funnybirds_dataset import PART_VARIANTS, concept_names, _FUNNYBIRDS_N_TRAIN
from src.probes.probe_utils import train_linear_probe, ProbeMetrics


# ---------------------------------------------------------------------------
# Layer discovery (same logic as train_probes_attributes.py)
# ---------------------------------------------------------------------------

def choose_layers(features_dir: Path) -> List[str]:
    """
    Auto-discover layers from <layer>_train.pt files that also have _test.pt.
    Returns layers in canonical order: conv1, layer1.0..layer4.2, avgpool.
    """
    layers = []
    for p in features_dir.glob("*_train.pt"):
        name = p.name.replace("_train.pt", "")
        if name == "labels":
            continue
        if (features_dir / f"{name}_test.pt").exists():
            layers.append(name)

    def sort_key(n: str):
        if n == "conv1":
            return (0, 0, 0)
        if n.startswith("layer"):
            if "." in n:
                stage, block = n.split(".", 1)
                return (int(stage.replace("layer", "")), 1, int(block))
            else:
                return (int(n.replace("layer", "")), 0, -1)
        if n == "avgpool":
            return (99, 0, 0)
        return (50, 0, 0)

    return sorted(set(layers), key=sort_key)


# ---------------------------------------------------------------------------
# FunnyBirds metadata loader
# ---------------------------------------------------------------------------

def load_funnybirds_labels(
    funnybirds_root: Path,
    img_ids_train: np.ndarray,
    img_ids_test: np.ndarray,
):
    """
    Load per-image concept and species labels aligned with img_ids_train/test.

    Returns:
      y_train_species, y_test_species : LongTensor  [N]  (0-based species id)
      y_train_concepts, y_test_concepts : dict {concept_name -> LongTensor [N]}
      cnames : list[str]  (26 concept names in canonical order)
    """
    metadata_dir = funnybirds_root / "metadata"

    # ── images.csv ────────────────────────────────────────────────────────────
    import pandas as pd
    images_df = pd.read_csv(metadata_dir / "images.csv")
    # columns: image_id, file_path, is_train, class_id
    img_to_species = dict(zip(images_df["image_id"], images_df["class_id"].astype(int)))

    def _species_labels(img_ids: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            [img_to_species.get(int(i), 0) for i in img_ids], dtype=torch.long
        )

    y_train_species = _species_labels(img_ids_train)
    y_test_species  = _species_labels(img_ids_test)

    # ── image_concepts_binary.csv ─────────────────────────────────────────────
    # Wide format: image_id, beak_0, beak_1, ..., tail_8
    concepts_df = pd.read_csv(metadata_dir / "image_concepts_binary.csv")
    concepts_df = concepts_df.set_index("image_id")

    cnames = concept_names()  # 26 names in canonical order

    def _concept_labels(img_ids: np.ndarray, col: str) -> torch.Tensor:
        return torch.tensor(
            [int(concepts_df.loc[int(i), col]) if int(i) in concepts_df.index else 0
             for i in img_ids],
            dtype=torch.long,
        )

    y_train_concepts: Dict[str, torch.Tensor] = {}
    y_test_concepts:  Dict[str, torch.Tensor] = {}
    for col in cnames:
        y_train_concepts[col] = _concept_labels(img_ids_train, col)
        y_test_concepts[col]  = _concept_labels(img_ids_test,  col)

    return (y_train_species, y_test_species,
            y_train_concepts, y_test_concepts,
            cnames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train layerwise linear probes on FunnyBirds features"
    )
    parser.add_argument(
        "--features_dir", type=str, required=True,
        help="Dir with {layer}_train.pt, {layer}_test.pt, labels_{train,test}.pt"
    )
    parser.add_argument(
        "--funnybirds_root", type=str, required=True,
        help="FunnyBirds root (must contain metadata/ from prepare_funnybirds_metadata.py)"
    )
    parser.add_argument(
        "--out_json", type=str,
        default="results/probes/resnet50_funnybirds_probes_fine.json",
        help="Output JSON path"
    )
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    fb_root  = Path(args.funnybirds_root)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover layers ────────────────────────────────────────────────────
    layers = choose_layers(feat_dir)
    if not layers:
        raise RuntimeError(
            f"No layers found in {feat_dir}. "
            "Expected files like layer3.2_train.pt. "
            "Run scripts/extract_features_funnybirds.py first."
        )
    print(f"[train_probes_funnybirds] Found {len(layers)} layers: {layers}")

    # ── 2. Load split order (image ids + species labels from saved .pt) ───────
    labels_train_pt = torch.load(feat_dir / "labels_train.pt", weights_only=True)
    labels_test_pt  = torch.load(feat_dir / "labels_test.pt",  weights_only=True)

    img_ids_train = labels_train_pt["image_ids"].numpy()
    img_ids_test  = labels_test_pt["image_ids"].numpy()

    print(f"[train_probes_funnybirds] Train images: {len(img_ids_train)}, "
          f"Test images: {len(img_ids_test)}")

    # ── 3. Load FunnyBirds concept + species labels ───────────────────────────
    metadata_dir = fb_root / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"{metadata_dir} not found. "
            "Run: python -m scripts.prepare_funnybirds_metadata "
            f"--funnybirds_root {fb_root}"
        )

    (y_train_species, y_test_species,
     y_train_concepts, y_test_concepts,
     cnames) = load_funnybirds_labels(fb_root, img_ids_train, img_ids_test)

    num_species = int(y_train_species.max().item() + 1)
    print(f"[train_probes_funnybirds] Species: {num_species}, Concepts: {len(cnames)}")

    # Concept name → part group (beak, eye, wing, foot, tail)
    # e.g. "beak_0" → "beak",  "tail_3" → "tail"
    concept_group: Dict[str, str] = {c: c.rsplit("_", 1)[0] for c in cnames}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_probes_funnybirds] Device: {device}")

    results = []

    # ── 4. Species probes (subclass) ─────────────────────────────────────────
    for layer_name in layers:
        print(f"[train_probes_funnybirds] Species probe — {layer_name}")
        x_train = torch.load(feat_dir / f"{layer_name}_train.pt", weights_only=True)
        x_test  = torch.load(feat_dir / f"{layer_name}_test.pt",  weights_only=True)

        metrics: ProbeMetrics = train_linear_probe(
            x_train=x_train, y_train=y_train_species,
            x_val=x_test,   y_val=y_test_species,
            num_classes=num_species,
            epochs=args.epochs, batch_size=args.batch_size, device=device,
        )

        results.append({
            "layer": layer_name,
            "target_type": "subclass",
            "target_name": "species",
            "group": "species",
            "train_acc":     metrics.train_acc,
            "val_acc":       metrics.val_acc,
            "best_val_acc":  metrics.best_val_acc,
        })

    # ── 5. Concept probes (attribute) ─────────────────────────────────────────
    for layer_name in layers:
        print(f"[train_probes_funnybirds] Concept probes — {layer_name}")
        x_train = torch.load(feat_dir / f"{layer_name}_train.pt", weights_only=True)
        x_test  = torch.load(feat_dir / f"{layer_name}_test.pt",  weights_only=True)

        for cname in cnames:
            metrics: ProbeMetrics = train_linear_probe(
                x_train=x_train, y_train=y_train_concepts[cname],
                x_val=x_test,   y_val=y_test_concepts[cname],
                num_classes=2,
                epochs=args.epochs, batch_size=args.batch_size, device=device,
            )

            results.append({
                "layer":         layer_name,
                "target_type":   "attribute",
                "target_name":   cname,
                "group":         concept_group[cname],
                "train_acc":     metrics.train_acc,
                "val_acc":       metrics.val_acc,
                "best_val_acc":  metrics.best_val_acc,
            })

    # ── 6. Save JSON ──────────────────────────────────────────────────────────
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    total_concept_rows = len(cnames) * len(layers)
    total_species_rows = len(layers)
    print(
        f"[train_probes_funnybirds] Wrote {len(results)} rows "
        f"({total_species_rows} species + {total_concept_rows} concept) "
        f"→ {out_path}"
    )


if __name__ == "__main__":
    main()