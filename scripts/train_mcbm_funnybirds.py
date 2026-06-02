#!/usr/bin/env python
"""
scripts/train_mcbm_funnybirds.py

Train a Minimal Concept Bottleneck Model (MCBM) on FunnyBirds.

Mirrors scripts/train_mcbm.py exactly in:
  - Architecture (MCBM class: backbone + concept_encoder + label_head)
  - Loss function (CE + lambda_c * BCE + gamma * IB penalty)
  - Two-stage training (backbone frozen, then joint fine-tune)
  - Checkpoint format (model_state_dict + config dict)
  - CLI argument names and defaults

Differences vs train_mcbm.py:
  - Uses FunnyBirdsDataset instead of CUBDataset
  - num_classes = 50 (FunnyBirds) instead of 200 (CUB)
  - num_concepts = 26 (FunnyBirds: 4+3+6+4+9 one-hot part variants)
  - No 10-90% prevalence filter (FunnyBirds concepts are perfectly balanced)
  - Default paths point to checkpoints_funnybirds/ and features_funnybirds/

Concepts (26 binary, one-hot over part variants, perfectly balanced):
    beak_0..beak_3  (4)   eye_0..eye_2  (3)   wing_0..wing_5  (6)
    foot_0..foot_3  (4)   tail_0..tail_8 (9)
    = 26 total

gamma sweep: 0.0, 0.1, 0.5, 1.0, 5.0  (same as CUB)
  gamma=0.0 -> standard CBM (sanity check)

Usage:
    python -m scripts.train_mcbm_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --checkpoint_dir  checkpoints_funnybirds \\
        --backbone_ckpt   checkpoints_funnybirds/resnet50_funnybirds_best.pth \\
        --gamma           0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from datasets.funnybirds_dataset import (
    FunnyBirdsDataset, NUM_CONCEPTS, concept_names
)


# ---------------------------------------------------------------------------
# Transforms (mirror train_resnet_funnybirds.py / train_mcbm.py exactly)
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
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(
    funnybirds_root: str,
    batch_size: int,
    num_workers: int = 4,
    concept_labels_path: str | None = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build train/val DataLoaders with FunnyBirds concept labels (26 binary).

    concept_labels_path: optional path to image-level concept labels (.npy)
        produced by scripts/make_image_level_concept_labels.py.
        When provided, the train loader uses per-image labels instead of
        the default species-level annotation-derived labels.

    Returns:
        train_loader, val_loader, num_classes
    """
    tf_train, tf_eval = _make_transforms()

    train_ds = FunnyBirdsDataset(
        funnybirds_root, split="train",
        transform=tf_train, include_concepts=True,
        concept_labels_path=concept_labels_path,
    )
    val_ds = FunnyBirdsDataset(
        funnybirds_root, split="test",
        transform=tf_eval, include_concepts=True,
    )

    n_classes = train_ds.num_classes()
    print(f"[MCBM-FB] {len(train_ds)} train, {len(val_ds)} test")
    print(f"[MCBM-FB] num_classes={n_classes}  num_concepts={NUM_CONCEPTS}")
    print(f"[MCBM-FB] concept_names: {concept_names()}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, n_classes


# ---------------------------------------------------------------------------
# Model (identical to train_mcbm.py MCBM class)
# ---------------------------------------------------------------------------

class MCBM(nn.Module):
    """
    Minimal Concept Bottleneck Model — identical to train_mcbm.py MCBM.

    For FunnyBirds: num_concepts=26, num_classes=50.
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes:  int   = 50,
        sigma:        float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma

        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.concept_encoder = nn.Linear(feat_dim, num_concepts)
        self.label_head       = nn.Linear(num_concepts, num_classes)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        z = self.concept_encoder(feats)

        if training and self.sigma > 0.0:
            z_sampled = z + self.sigma * torch.randn_like(z)
        else:
            z_sampled = z

        y_logits = self.label_head(z_sampled)
        return y_logits, z, z_sampled

    @staticmethod
    def q_phi(z: torch.Tensor) -> torch.Tensor:
        return 6.0 * torch.sigmoid(z) - 3.0

    @staticmethod
    def ib_penalty(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Pull each z_j toward +3 (concept present) or -3 (concept absent).
        # This matches the paper's ||z_j - g(c_j)||^2 formulation.
        targets = 3.0 * (2.0 * c - 1.0)   # c=1 → +3,  c=0 → -3
        return ((z - targets) ** 2).mean()


# ---------------------------------------------------------------------------
# Training / evaluation (identical to train_mcbm.py)
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
    if freeze_backbone:
        model.backbone.eval()
        model.concept_encoder.train()
        model.label_head.train()
    else:
        model.train()

    ce_fn  = nn.CrossEntropyLoss()
    bce_fn = nn.BCEWithLogitsLoss()

    total_sum = c_sum = task_sum = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        y    = batch["label"].to(device)
        c    = batch["concepts"].to(device)

        if freeze_backbone:
            with torch.no_grad():
                feats = model.backbone(imgs)
            z = model.concept_encoder(feats)
            z_s = z + model.sigma * torch.randn_like(z) if model.sigma > 0.0 else z
            y_logits = model.label_head(z_s)
        else:
            y_logits, z, z_s = model(imgs, training=True)

        task_loss = ce_fn(y_logits, y)
        # Concept loss and IB penalty use clean z, not noisy z_s.
        # Noise flows only to the label head to enforce the information bottleneck.
        c_loss    = bce_fn(z, c)
        ib_loss   = model.ib_penalty(z, c)

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
    model.eval()

    ce_fn  = nn.CrossEntropyLoss()
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
        ib_loss   = model.ib_penalty(z, c)
        loss      = task_loss + lambda_c * c_loss + gamma * ib_loss
        total_sum += loss.item()

        c_preds    = (torch.sigmoid(z) > 0.5).float()
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
    parser = argparse.ArgumentParser(
        description="Train MCBM on FunnyBirds (50 classes, 26 concepts)"
    )
    parser.add_argument("--funnybirds_root", type=str, required=True,
                        help="Path to FunnyBirds root directory")
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints_funnybirds",
                        help="Directory to save checkpoints")
    parser.add_argument("--backbone_ckpt",   type=str,
                        default="checkpoints_funnybirds/resnet50_funnybirds_best.pth",
                        help="Path to pretrained ResNet-50 checkpoint (from train_resnet_funnybirds.py)")
    parser.add_argument("--gamma",           type=float, default=0.1,
                        help="IB penalty weight. gamma=0.0 recovers standard CBM.")
    parser.add_argument("--epochs_stage1",   type=int, default=12,
                        help="Epochs with backbone frozen")
    parser.add_argument("--epochs_stage2",   type=int, default=10,
                        help="Epochs with full model unfrozen")
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--sigma",           type=float, default=1.0,
                        help="Noise std for stochastic z. sigma=0 disables noise.")
    parser.add_argument("--lambda_c",        type=float, default=1.0,
                        help="Weight for concept prediction loss (BCE term).")
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--device",          type=str,   default="cuda")
    parser.add_argument("--concept_labels",  type=str,   default=None,
                        help="Path to image-level concept labels .npy "
                             "(from scripts/make_image_level_concept_labels.py). "
                             "When provided, replaces species-level labels for the train set.")
    parser.add_argument("--ckpt_suffix",     type=str,   default="",
                        help="Extra suffix appended to checkpoint filename before .pth. "
                             "Use to save side-by-side models, e.g. '_fix' → mcbm_fb_gamma0.1_fix.pth.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[MCBM-FB] Using device: {device}")
    print(f"[MCBM-FB] gamma={args.gamma}  sigma={args.sigma}  lambda_c={args.lambda_c}")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_tag = "_rl" if args.concept_labels else ""
    label_tag += args.ckpt_suffix
    if args.concept_labels:
        print(f"[MCBM-FB] Image-level concept labels: {args.concept_labels}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, n_classes = build_loaders(
        args.funnybirds_root, args.batch_size, args.num_workers,
        concept_labels_path=args.concept_labels,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MCBM(num_concepts=NUM_CONCEPTS, num_classes=n_classes, sigma=args.sigma)

    # Initialise backbone from pre-trained FunnyBirds ResNet-50
    state = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=True)
    # The checkpoint may have "fc.*" keys (n_classes=50 head); strip them — we use our own
    backbone_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[MCBM-FB] Backbone init from {args.backbone_ckpt}")
    if missing:
        print(f"[MCBM-FB] missing keys: {missing}")
    if unexpected:
        print(f"[MCBM-FB] unexpected keys: {unexpected}")

    model.to(device)

    # ── Stage 1: backbone frozen ───────────────────────────────────────────────
    print(f"\n[MCBM-FB] Stage 1: backbone frozen, {args.epochs_stage1} epochs")
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
    print(f"\n[MCBM-FB] Stage 2: full fine-tune, {args.epochs_stage2} epochs")
    for p in model.backbone.parameters():
        p.requires_grad = True

    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
    # label_tag = "_rl" when trained on image-level (relabeled) concepts
    ckpt_path = out_dir / f"mcbm_fb_gamma{args.gamma}{label_tag}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_concepts":       NUM_CONCEPTS,
                "num_classes":        n_classes,
                "sigma":              args.sigma,
                "gamma":              args.gamma,
                "lambda_c":           args.lambda_c,
                "concept_names":      concept_names(),
                "dataset":            "funnybirds",
                "concept_labels_tag": (label_tag or "species_level"),
                "ckpt_suffix":        args.ckpt_suffix,
            },
        },
        ckpt_path,
    )
    print(f"\n[MCBM-FB] Saved checkpoint to {ckpt_path}")
    print(f"[MCBM-FB] num_concepts={NUM_CONCEPTS}  best_stage1_val_c_acc={best_stage1_c_acc:.4f}")


if __name__ == "__main__":
    main()
