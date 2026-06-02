#!/usr/bin/env python
"""
scripts/extract_features_mcbm.py

Extract intermediate ResNet-50 backbone features from a trained MCBM checkpoint.

Mirrors scripts/extract_features.py exactly in terms of:
  - probe points (same 18 fine-grained layers)
  - hook logic and GAP reduction
  - output file naming: {layer_name}_{split}.pt  +  labels_{split}.pt

The MCBM checkpoint stores the full model (backbone + concept_encoder + label_head).
We load only the backbone weights for feature extraction.

Usage:
    python scripts/extract_features_mcbm.py \
        --cub_root data/CUB_200_2011 \
        --checkpoint checkpoints_mcbm/mcbm_gamma0.1.pth \
        --features_dir features/resnet50_mcbm_gamma0.1 \
        --batch_size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from datasets.cub_dataset import CUBDataset


# ---------------------------------------------------------------------------
# Transforms (eval only; mirror extract_features.py exactly)
# ---------------------------------------------------------------------------

def _make_eval_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Layer selection (identical to extract_features.py get_feature_layers)
# ---------------------------------------------------------------------------

def get_feature_layers(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    Fine-grained probe points:
      conv1 (after maxpool), each Bottleneck block output, and avgpool.
    Identical to scripts/extract_features.py.
    """
    layers = {}

    # conv1: use maxpool output
    layers["conv1"] = model.maxpool

    # each residual block output
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        seq = getattr(model, stage)
        for b, block in enumerate(seq):
            layers[f"{stage}.{b}"] = block

    layers["avgpool"] = model.avgpool
    return layers


# ---------------------------------------------------------------------------
# Feature extraction (identical logic to extract_features.py)
# ---------------------------------------------------------------------------

def extract_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    split_name: str,
):
    """
    Registers forward hooks on the 18 fine-grained probe points,
    runs inference, and saves one .pt file per layer plus labels_{split}.pt.
    Mirrors extract_features.py extract_split() exactly.
    """
    feature_layers = get_feature_layers(model)
    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in feature_layers}
    labels_list: List[torch.Tensor] = []
    image_ids_list: List[torch.Tensor] = []

    hooks = []

    def make_hook(layer_name: str):
        def hook(module, inp, out):
            # out: (B,C,H,W) or (B,C) or (B,C,1,1)
            if out.dim() == 4:
                feat = out.mean(dim=(2, 3))  # GAP -> (B,C)
            elif out.dim() == 2:
                feat = out
            else:
                feat = out.view(out.size(0), -1)
            activations[layer_name].append(feat.detach().cpu())
        return hook

    for name, module in feature_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {split_name}"):
            imgs      = batch["image"].to(device)
            labels    = batch["label"]
            image_ids = batch["image_id"]

            _ = model(imgs)

            labels_list.append(labels.cpu())
            image_ids_list.append(
                image_ids.cpu()
                if isinstance(image_ids, torch.Tensor)
                else torch.tensor(image_ids)
            )

    for h in hooks:
        h.remove()

    out_dir.mkdir(parents=True, exist_ok=True)

    labels_tensor    = torch.cat(labels_list,    dim=0)
    image_ids_tensor = torch.cat(image_ids_list, dim=0)
    torch.save(
        {"labels": labels_tensor, "image_ids": image_ids_tensor},
        out_dir / f"labels_{split_name}.pt",
    )

    for name, tensor_list in activations.items():
        feats = torch.cat(tensor_list, dim=0)
        torch.save(feats, out_dir / f"{name}_{split_name}.pt")
        print(
            f"[extract_features_mcbm] Saved {name}_{split_name}.pt "
            f"with shape {feats.shape}"
        )


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_mcbm_backbone(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a MCBM checkpoint and returns the ResNet-50 backbone only.

    The MCBM checkpoint has structure:
      {
        "model_state_dict": { "backbone.*": ..., "concept_encoder.*": ..., "label_head.*": ... },
        "config": { ... }
      }

    We extract only the "backbone.*" keys, strip the prefix, and load into
    a standard ResNet-50 with fc=Identity().
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Support both plain state dict and nested dict saved by train_mcbm.py
    if "model_state_dict" in ckpt:
        full_state = ckpt["model_state_dict"]
    else:
        full_state = ckpt

    # Extract backbone weights only
    prefix = "backbone."
    backbone_state = {
        k[len(prefix):]: v
        for k, v in full_state.items()
        if k.startswith(prefix)
    }

    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[extract_features_mcbm] Backbone missing keys: {missing}")
    if unexpected:
        print(f"[extract_features_mcbm] Backbone unexpected keys: {unexpected}")

    return backbone.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract MCBM backbone features for recall gap analysis"
    )
    parser.add_argument("--cub_root",     type=str, required=True,
                        help="Path to CUB_200_2011 root directory")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to MCBM checkpoint (.pth)")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Output directory for extracted features")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_features_mcbm] Using device: {device}")
    print(f"[extract_features_mcbm] Checkpoint: {args.checkpoint}")
    print(f"[extract_features_mcbm] Output dir: {args.features_dir}")

    tf_eval = _make_eval_transform()

    train_ds = CUBDataset(args.cub_root, split="train", transform=tf_eval)
    test_ds  = CUBDataset(args.cub_root, split="test",  transform=tf_eval)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    backbone = load_mcbm_backbone(args.checkpoint, device)
    print(f"[extract_features_mcbm] Backbone loaded from {args.checkpoint}")

    out_dir = Path(args.features_dir)
    extract_split(backbone, train_loader, device, out_dir, "train")
    extract_split(backbone, test_loader,  device, out_dir, "test")

    print(f"[extract_features_mcbm] Done. Features saved to {out_dir}")


if __name__ == "__main__":
    main()
