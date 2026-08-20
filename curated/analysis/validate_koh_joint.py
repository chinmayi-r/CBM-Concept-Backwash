#!/usr/bin/env python3
"""Validate that a checkpoint is the requested official Koh Joint CBM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--koh-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True,
                        choices=("funnybirds", "cub70", "cub"))
    parser.add_argument("--labels", required=True,
                        choices=("standard", "rlv2"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--num-attributes", required=True, type=int)
    parser.add_argument("--backbone", choices=("inception_v3", "resnet50"),
                        default="inception_v3")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--status", choices=("SUCCESS", "INCOMPLETE"),
                        default="SUCCESS")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    if args.dataset != "funnybirds" and args.labels != "standard":
        raise SystemExit("CUB/CUB70 RLv2 is not defined")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        raise SystemExit(f"missing/empty checkpoint: {args.checkpoint}")
    if not (args.koh_root / "experiments.py").is_file():
        raise SystemExit(f"not an official Koh checkout: {args.koh_root}")

    curated = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(curated / "compat"))
    sys.path.insert(0, str(args.koh_root))
    try:
        model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(args.checkpoint, map_location="cpu")
    if type(model).__name__ != "End2EndModel":
        raise SystemExit(
            f"wrong framework/model: expected Koh End2EndModel, got {type(model).__name__}"
        )
    if getattr(model, "use_sigmoid", None):
        raise SystemExit("wrong Koh variant: paper Joint model must read raw logits")
    if getattr(model, "use_relu", None):
        raise SystemExit("wrong Koh variant: class head must read raw logits without ReLU")
    if args.backbone == "resnet50":
        if getattr(model, "curated_framework", None) != "koh_joint":
            raise SystemExit("ResNet checkpoint lacks Koh Joint framework marker")
        if getattr(model, "curated_backbone", None) != "resnet50":
            raise SystemExit("checkpoint lacks ResNet-50 backbone marker")
        first = getattr(model, "first_model", None)
        if type(first).__name__ != "KohResNet50ConceptEncoder":
            raise SystemExit(
                f"wrong ResNet image encoder: {type(first).__name__ if first else None}"
            )
        if len(getattr(first, "main_heads", ())) != args.num_attributes:
            raise SystemExit("ResNet concept-head count mismatch")

    state = model.state_dict()
    bad = [name for name, value in state.items()
           if torch.is_floating_point(value) and not torch.isfinite(value).all()]
    if bad:
        raise SystemExit(f"non-finite checkpoint tensors: {bad[:5]}")

    # Koh Joint wraps the image-to-concept model and the linear c->y model in
    # End2EndModel. The latter must map exactly n_attributes -> n_classes.
    candidates = [(name, value) for name, value in state.items()
                  if name.endswith("weight") and value.ndim == 2
                  and tuple(value.shape) == (args.num_classes, args.num_attributes)]
    if not candidates:
        shapes = {name: tuple(value.shape) for name, value in state.items()
                  if name.endswith("weight") and value.ndim == 2}
        raise SystemExit(
            "class-head shape mismatch: expected "
            f"({args.num_classes}, {args.num_attributes}); found {shapes}"
        )

    manifest = {
        "status": args.status,
        "framework": "official_koh_conceptbottleneck",
        "model": "Joint",
        "backbone": args.backbone,
        "use_sigmoid": False,
        "attr_loss_weight": 0.01,
        "dataset": args.dataset,
        "labels": args.labels,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "num_attributes": args.num_attributes,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "class_head_candidates": [name for name, _ in candidates],
        "note": args.note,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[KOH JOINT CHECKPOINT PASS] status={args.status} {args.manifest}")


if __name__ == "__main__":
    main()
