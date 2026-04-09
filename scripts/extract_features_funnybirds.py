#!/usr/bin/env python
"""
scripts/extract_features_funnybirds.py

Extract intermediate ResNet-50 backbone features from a trained MCBM-FunnyBirds
checkpoint (or a plain FunnyBirds ResNet checkpoint).

Mirrors scripts/extract_features_mcbm.py exactly in:
  - Probe points (same 18 fine-grained layers)
  - Hook logic and GAP reduction
  - Output file naming: {layer_name}_{split}.pt  +  labels_{split}.pt

The only difference is using FunnyBirdsDataset instead of CUBDataset.

Usage:
    # MCBM checkpoint (recommended for entanglement analysis):
    python -m scripts.extract_features_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --checkpoint      checkpoints_funnybirds/mcbm_fb_gamma0.1.pth \\
        --features_dir    features/resnet50_mcbm_fb_gamma0.1

    # Plain ResNet checkpoint (for sanity check / stage-1 probe):
    python -m scripts.extract_features_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --checkpoint      checkpoints_funnybirds/resnet50_funnybirds_best.pth \\
        --features_dir    features/resnet50_funnybirds \\
        --plain_resnet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from datasets.funnybirds_dataset import FunnyBirdsDataset


# ---------------------------------------------------------------------------
# Transforms (eval only; mirror extract_features_mcbm.py)
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
# Layer selection (identical to extract_features_mcbm.py)
# ---------------------------------------------------------------------------

def get_feature_layers(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    Fine-grained probe points: same 18 layers as CUB pipeline.
    """
    layers = {}
    layers["conv1"] = model.maxpool
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        seq = getattr(model, stage)
        for b, block in enumerate(seq):
            layers[f"{stage}.{b}"] = block
    layers["avgpool"] = model.avgpool
    return layers


# ---------------------------------------------------------------------------
# Feature extraction (identical to extract_features_mcbm.py)
# ---------------------------------------------------------------------------

def extract_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    split_name: str,
):
    feature_layers = get_feature_layers(model)
    activations: Dict[str, List[torch.Tensor]] = {n: [] for n in feature_layers}
    labels_list: List[torch.Tensor]     = []
    image_ids_list: List[torch.Tensor]  = []

    hooks = []

    def make_hook(layer_name: str):
        def hook(module, inp, out):
            if out.dim() == 4:
                feat = out.mean(dim=(2, 3))
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
        print(f"[extract_fb] Saved {name}_{split_name}.pt  shape={feats.shape}")


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_backbone(ckpt_path: str, device: torch.device, plain_resnet: bool) -> torch.nn.Module:
    """
    Load backbone from either:
      - MCBM checkpoint (model_state_dict with "backbone.*" keys)
      - Plain ResNet checkpoint (state dict with standard ResNet keys + "fc.*")
    Returns ResNet-50 with fc=Identity().
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if plain_resnet:
        # State dict from train_resnet_funnybirds.py: standard ResNet keys
        full_state = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt else ckpt["model_state_dict"]
        backbone_state = {k: v for k, v in full_state.items() if not k.startswith("fc.")}
    else:
        # MCBM checkpoint: nested under "model_state_dict", keys prefixed "backbone."
        full_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
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
        print(f"[extract_fb] Backbone missing keys: {missing}")
    if unexpected:
        print(f"[extract_fb] Backbone unexpected keys: {unexpected}")

    return backbone.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract FunnyBirds MCBM backbone features"
    )
    parser.add_argument("--funnybirds_root", type=str, required=True,
                        help="Path to FunnyBirds root directory")
    parser.add_argument("--checkpoint",      type=str, required=True,
                        help="Path to MCBM or plain-ResNet checkpoint (.pth)")
    parser.add_argument("--features_dir",    type=str, required=True,
                        help="Output directory for extracted features")
    parser.add_argument("--plain_resnet",    action="store_true",
                        help="Set if checkpoint is from train_resnet_funnybirds.py "
                             "(not an MCBM checkpoint)")
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--num_workers",     type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_fb] Using device:   {device}")
    print(f"[extract_fb] Checkpoint:     {args.checkpoint}")
    print(f"[extract_fb] Output dir:     {args.features_dir}")
    print(f"[extract_fb] plain_resnet:   {args.plain_resnet}")

    tf_eval = _make_eval_transform()
    train_ds = FunnyBirdsDataset(args.funnybirds_root, split="train", transform=tf_eval)
    test_ds  = FunnyBirdsDataset(args.funnybirds_root, split="test",  transform=tf_eval)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    backbone = load_backbone(args.checkpoint, device, plain_resnet=args.plain_resnet)
    print(f"[extract_fb] Backbone loaded from {args.checkpoint}")

    out_dir = Path(args.features_dir)
    extract_split(backbone, train_loader, device, out_dir, "train")
    extract_split(backbone, test_loader,  device, out_dir, "test")

    print(f"[extract_fb] Done. Features saved to {out_dir}")


if __name__ == "__main__":
    main()
