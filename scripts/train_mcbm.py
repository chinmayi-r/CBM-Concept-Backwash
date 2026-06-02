#!/usr/bin/env python
"""
scripts/train_mcbm.py

Trains a Minimal Concept Bottleneck Model (MCBM) on CUB-200-2011.

Reference:
  "There Was Never a Bottleneck in Concept Bottleneck Models"
  Almudévar et al., arXiv:2506.04877, 2025
  Official repo: https://github.com/antonioalmudevar/minimal_cbm

Architecture (ResNet-50 adaptation for our pipeline):
  - Backbone: ResNet-50 (initialized from resnet50_cub_best.pth, fc=Identity)
  - Concept encoder: Linear(2048, num_concepts) -> z in R^num_concepts
    Each z_j is a scalar; at train time z += sigma * randn (stochastic encoder)
  - Label head: Linear(num_concepts, 200) -> species logits  (takes z as input)
  - q_phi(z): fixed formula  6 * sigmoid(z) - 3  (no learned params;
    matches the official MCBM implementation)

Loss:
  L = CE(y_hat, y)
    + lambda_c * BCE(z, c)                          [concept loss]
    + gamma   * 0.2 * mean_j( (6*sigmoid(z_j)-3 - z_j)^2 )  [IB penalty]

  The IB term is the MSE approximation to KL(p_theta(z|x) || q_phi(z|c))
  for equal-variance Gaussians, matching the official implementation.

Training:
  Stage 1 (backbone frozen): train concept encoder + label head
  Stage 2 (backbone unfrozen): train all jointly

Output:
  checkpoints_mcbm/mcbm_gamma{gamma}.pth
  containing model_state_dict + config dict

gamma sweep recommended: 0.0, 0.1, 0.5, 1.0, 5.0
  gamma=0.0 recovers standard CBM (sanity check vs cbm_species.csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from datasets.cub_dataset import CUBDataset
from datasets.cub_metadata import load_cub_metadata


# ---------------------------------------------------------------------------
# Transforms (mirror train_resnet.py exactly)
# ---------------------------------------------------------------------------

def _make_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    tf_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return tf_train, tf_eval


# ---------------------------------------------------------------------------
# Dataset with attribute concepts
# ---------------------------------------------------------------------------

class CUBWithAttributes(Dataset):
    """
    Thin wrapper around CUBDataset that also returns binary attribute labels.

    Each item adds:
      "concepts": float32 tensor of shape (num_concepts,) with binary attribute values
    """

    def __init__(
        self,
        cub_root: str,
        split: str,
        chosen_attr_cols: List[str],
        img_attr_map: Dict[int, torch.Tensor],
        transform=None,
    ):
        self.inner = CUBDataset(cub_root, split=split, transform=transform)
        self.chosen_attr_cols = chosen_attr_cols
        self.img_attr_map = img_attr_map
        self.num_concepts = len(chosen_attr_cols)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        sample = self.inner[idx]
        img_id = int(sample["image_id"])
        concepts = self.img_attr_map.get(
            img_id,
            torch.zeros(self.num_concepts, dtype=torch.float32),
        )
        sample["concepts"] = concepts
        return sample


def build_loaders(
    cub_root: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, List[str], int]:
    """
    Build train/val DataLoaders with attribute concept labels.

    Attributes are filtered to those with 10%-90% prevalence in the training
    set, matching the existing CBM training pipeline.

    Returns:
        train_loader, val_loader, chosen_attr_cols, num_concepts
    """
    tf_train, tf_eval = _make_transforms()
    meta = load_cub_metadata(cub_root)

    img_attr_binary = meta.image_attributes_binary
    if img_attr_binary is None:
        raise RuntimeError(
            "Attribute labels not found. "
            "Expected <cub_root>/attributes/image_attribute_labels.txt"
        )

    img_attr_binary = img_attr_binary.set_index("image_id")
    attr_cols = [c for c in img_attr_binary.columns if c.startswith("attr_")]

    # Filter to attributes with 10%-90% prevalence in training images
    train_image_ids = meta.images[meta.images["is_train"] == 1]["image_id"]
    train_attr = img_attr_binary.loc[
        img_attr_binary.index.isin(train_image_ids), attr_cols
    ]
    freqs = train_attr.mean(axis=0)
    valid_mask = (freqs > 0.10) & (freqs < 0.90)
    chosen_attr_cols = list(freqs[valid_mask].index)
    num_concepts = len(chosen_attr_cols)
    print(f"[MCBM] Using {num_concepts} attribute concepts")

    # Build image_id -> concept tensor mapping (covers all images)
    img_attr_map: Dict[int, torch.Tensor] = {}
    for img_id, row in img_attr_binary.iterrows():
        vec = torch.tensor(
            row[chosen_attr_cols].values.astype(np.float32),
            dtype=torch.float32,
        )
        img_attr_map[int(img_id)] = vec

    train_ds = CUBWithAttributes(
        cub_root, "train", chosen_attr_cols, img_attr_map, tf_train
    )
    val_ds = CUBWithAttributes(
        cub_root, "test", chosen_attr_cols, img_attr_map, tf_eval
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, chosen_attr_cols, num_concepts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MCBM(nn.Module):
    """
    Minimal Concept Bottleneck Model (ResNet-50 backbone).

    Architecture:
      backbone: ResNet-50 (fc=Identity) -> 2048-dim avgpool features
      concept_encoder: Linear(2048, num_concepts) -> z in R^num_concepts
      label_head: Linear(num_concepts, 200) -> species logits

    Stochastic encoder (train time only):
      z_sampled = z + sigma * randn_like(z)

    Representation head (fixed, no learned params):
      q_phi(z) = 6 * sigmoid(z) - 3
      Maps z ≈ 0 (concept absent boundary) -> -3,
           z ≈ 0 (concept present boundary) -> +3.

    IB penalty:
      ib_loss = 0.2 * mean( (q_phi(z) - z)^2 )
      = MSE approximation to KL(p_theta(z|x) || q_phi(z|c))
        for equal-variance Gaussians (matches official MCBM repo).
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int = 200,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma

        # ResNet-50 backbone with identity fc
        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Per-concept scalar encoder: 2048 -> num_concepts
        self.concept_encoder = nn.Linear(feat_dim, num_concepts)

        # Label head: num_concepts -> 200 species
        self.label_head = nn.Linear(num_concepts, num_classes)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            y_logits: (B, num_classes)
            z:        (B, num_concepts)  -- clean concept logits (always)
            z_sampled: (B, num_concepts) -- noisy at train time, == z at eval
        """
        feats = self.backbone(x)             # (B, 2048)
        z = self.concept_encoder(feats)       # (B, num_concepts)

        if training and self.sigma > 0.0:
            z_sampled = z + self.sigma * torch.randn_like(z)
        else:
            z_sampled = z

        y_logits = self.label_head(z_sampled)  # (B, num_classes)
        return y_logits, z, z_sampled

    @staticmethod
    def q_phi(z: torch.Tensor) -> torch.Tensor:
        """Fixed representation head: q_phi(c) = 6 * sigmoid(z) - 3."""
        return 6.0 * torch.sigmoid(z) - 3.0

    @staticmethod
    def ib_penalty(z: torch.Tensor) -> torch.Tensor:
        """
        IB penalty term (scalar).
          0.2 * mean_over_all_j( (q_phi(z_j) - z_j)^2 )
        Measures how far z is from the concept-explainable representation.
        gamma=0 disables this term entirely in the loss (no multiply needed here).
        """
        q = MCBM.q_phi(z)
        return 0.2 * ((q - z) ** 2).mean()


# ---------------------------------------------------------------------------
# Training / evaluation utilities
# ---------------------------------------------------------------------------

def _train_epoch(
    model: MCBM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gamma: float,
    lambda_c: float,
    freeze_backbone: bool,
) -> Tuple[float, float, float]:
    """One training epoch. Returns (total_loss, concept_loss, task_loss)."""

    if freeze_backbone:
        model.backbone.eval()       # freeze BN statistics
        model.concept_encoder.train()
        model.label_head.train()
    else:
        model.train()

    ce_fn = nn.CrossEntropyLoss()
    bce_fn = nn.BCEWithLogitsLoss()

    total_sum = c_sum = task_sum = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        y    = batch["label"].to(device)          # (B,)  long
        c    = batch["concepts"].to(device)       # (B, num_concepts) float

        if freeze_backbone:
            with torch.no_grad():
                feats = model.backbone(imgs)
            z = model.concept_encoder(feats)
            if model.sigma > 0.0:
                z_s = z + model.sigma * torch.randn_like(z)
            else:
                z_s = z
            y_logits = model.label_head(z_s)
        else:
            y_logits, z, z_s = model(imgs, training=True)

        task_loss = ce_fn(y_logits, y)
        # Concept loss and IB penalty use clean z (not noisy z_s).
        # Noise is only for the label head to enforce the information bottleneck.
        c_loss    = bce_fn(z, c)
        ib_loss   = model.ib_penalty(z)

        loss = task_loss + lambda_c * c_loss + gamma * ib_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_sum += loss.item()
        c_sum     += c_loss.item()
        task_sum  += task_loss.item()
        n += 1

    return total_sum / n, c_sum / n, task_sum / n


@torch.no_grad()
def _eval_epoch(
    model: MCBM,
    loader: DataLoader,
    device: torch.device,
    gamma: float,
    lambda_c: float,
) -> Tuple[float, float, float]:
    """One eval pass. Returns (total_loss, concept_acc, task_acc)."""
    model.eval()

    ce_fn = nn.CrossEntropyLoss()
    bce_fn = nn.BCEWithLogitsLoss()

    total_sum = 0.0
    c_correct = c_total = 0
    t_correct = t_total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        imgs = batch["image"].to(device)
        y    = batch["label"].to(device)
        c    = batch["concepts"].to(device)

        y_logits, z, z_s = model(imgs, training=False)

        task_loss = ce_fn(y_logits, y)
        c_loss    = bce_fn(z, c)
        ib_loss   = model.ib_penalty(z)
        loss      = task_loss + lambda_c * c_loss + gamma * ib_loss
        total_sum += loss.item()

        c_preds = (torch.sigmoid(z) > 0.5).float()
        c_correct += (c_preds == c).sum().item()
        c_total   += c.numel()

        t_correct += (y_logits.argmax(dim=1) == y).sum().item()
        t_total   += y.size(0)

    n = len(loader)
    return (
        total_sum / n,
        c_correct / c_total,
        t_correct / t_total,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MCBM on CUB-200-2011")
    parser.add_argument("--cub_root",       type=str, required=True,
                        help="Path to CUB_200_2011 root directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_mcbm",
                        help="Directory to save checkpoints")
    parser.add_argument("--backbone_ckpt",  type=str,
                        default="checkpoints/resnet50_cub_best.pth",
                        help="Path to pretrained ResNet-50 checkpoint")
    parser.add_argument("--gamma",          type=float, default=0.1,
                        help="IB penalty weight. gamma=0.0 recovers standard CBM.")
    parser.add_argument("--epochs_stage1",  type=int, default=12,
                        help="Epochs with backbone frozen (concept+label heads only)")
    parser.add_argument("--epochs_stage2",  type=int, default=10,
                        help="Epochs with full model unfrozen (joint fine-tune)")
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--sigma",          type=float, default=1.0,
                        help="Noise std for stochastic z. sigma=0 disables noise.")
    parser.add_argument("--lambda_c",       type=float, default=1.0,
                        help="Weight for concept prediction loss (BCE term).")
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--device",         type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[MCBM] Using device: {device}")
    print(f"[MCBM] gamma={args.gamma}  sigma={args.sigma}  lambda_c={args.lambda_c}")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, chosen_attr_cols, num_concepts = build_loaders(
        args.cub_root, args.batch_size, args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MCBM(num_concepts=num_concepts, num_classes=200, sigma=args.sigma)

    # Initialise backbone from pretrained ResNet-50 checkpoint
    state = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"[MCBM] Backbone init from {args.backbone_ckpt}")
    if missing:
        print(f"[MCBM] load_state_dict missing keys: {missing}")
    if unexpected:
        print(f"[MCBM] load_state_dict unexpected keys: {unexpected}")

    model.to(device)

    # ── Stage 1: backbone frozen ───────────────────────────────────────────────
    print(f"\n[MCBM] Stage 1: backbone frozen, {args.epochs_stage1} epochs")
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt1 = torch.optim.AdamW(
        list(model.concept_encoder.parameters()) +
        list(model.label_head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    best_stage1_c_acc = 0.0
    for epoch in range(1, args.epochs_stage1 + 1):
        tr_loss, tr_c_loss, tr_task = _train_epoch(
            model, train_loader, opt1, device,
            gamma=args.gamma, lambda_c=args.lambda_c, freeze_backbone=True,
        )
        val_loss, val_c_acc, val_task_acc = _eval_epoch(
            model, val_loader, device, gamma=args.gamma, lambda_c=args.lambda_c,
        )
        print(
            f"[Concepts][{epoch:2d}]  "
            f"tr_loss={tr_loss:.4f}  tr_c_loss={tr_c_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_c_acc={val_c_acc:.4f}  "
            f"val_task_acc={val_task_acc:.4f}"
        )
        best_stage1_c_acc = max(best_stage1_c_acc, val_c_acc)

    # ── Stage 2: full fine-tune ────────────────────────────────────────────────
    print(f"\n[MCBM] Stage 2: full fine-tune, {args.epochs_stage2} epochs")
    for p in model.backbone.parameters():
        p.requires_grad = True

    opt2 = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs_stage2 + 1):
        tr_loss, tr_c_loss, tr_task = _train_epoch(
            model, train_loader, opt2, device,
            gamma=args.gamma, lambda_c=args.lambda_c, freeze_backbone=False,
        )
        val_loss, val_c_acc, val_task_acc = _eval_epoch(
            model, val_loader, device, gamma=args.gamma, lambda_c=args.lambda_c,
        )
        print(
            f"[Labels][{epoch:2d}]  "
            f"tr_loss={tr_loss:.4f}  tr_task={tr_task:.4f}  "
            f"val_loss={val_loss:.4f}  val_c_acc={val_c_acc:.4f}  "
            f"val_task_acc={val_task_acc:.4f}"
        )

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = out_dir / f"mcbm_gamma{args.gamma}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_concepts": num_concepts,
                "num_classes":  200,
                "sigma":        args.sigma,
                "gamma":        args.gamma,
                "lambda_c":     args.lambda_c,
                "attr_cols":    chosen_attr_cols,
            },
        },
        ckpt_path,
    )
    print(f"\n[MCBM] Saved checkpoint to {ckpt_path}")
    print(
        f"[MCBM] num_concepts={num_concepts}  "
        f"best_stage1_val_c_acc={best_stage1_c_acc:.4f}"
    )


if __name__ == "__main__":
    main()
