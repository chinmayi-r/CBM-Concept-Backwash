#!/usr/bin/env python
"""
scripts/train_cbm_funnybirds.py

Train a standard Concept Bottleneck Model (CBM) on FunnyBirds.

Mirrors train_mcbm_funnybirds.py exactly EXCEPT:
  - No IB penalty (no gamma, no q_phi, no reparameterization noise)
  - Concept bottleneck uses sigmoid(c_logits) directly (not stochastic z)
  - Saved to checkpoints_funnybirds/cbm_funnybirds.pth

Checkpoint format matches MCBM (model_state_dict with backbone.* keys) so
extract_features_funnybirds.py works unchanged (without --plain_resnet).

Usage:
    python -m scripts.train_cbm_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --checkpoint_dir  checkpoints_funnybirds \\
        --backbone_ckpt   checkpoints_funnybirds/resnet50_funnybirds_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from datasets.funnybirds_dataset import (
    FunnyBirdsDataset, NUM_CONCEPTS, concept_names
)


# ---------------------------------------------------------------------------
# Transforms (identical to train_mcbm_funnybirds.py)
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
) -> Tuple[DataLoader, DataLoader, int]:
    tf_train, tf_eval = _make_transforms()

    train_ds = FunnyBirdsDataset(
        funnybirds_root, split="train",
        transform=tf_train, include_concepts=True,
    )
    val_ds = FunnyBirdsDataset(
        funnybirds_root, split="test",
        transform=tf_eval, include_concepts=True,
    )

    n_classes = train_ds.num_classes()
    print(f"[CBM-FB] {len(train_ds)} train, {len(val_ds)} test")
    print(f"[CBM-FB] num_classes={n_classes}  num_concepts={NUM_CONCEPTS}")

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
# Model
# ---------------------------------------------------------------------------

class CBMFunnyBirds(nn.Module):
    """
    Standard Concept Bottleneck Model for FunnyBirds.

    backbone  → concept_head → sigmoid → label_head
    [2048]       [26]                      [50]

    No reparameterization noise, no IB penalty.
    """

    def __init__(self, num_concepts: int, num_classes: int = 50):
        super().__init__()
        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.concept_head = nn.Linear(feat_dim, num_concepts)
        self.label_head   = nn.Linear(num_concepts, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats    = self.backbone(x)
        c_logits = self.concept_head(feats)
        y_logits = self.label_head(torch.sigmoid(c_logits))
        return y_logits, c_logits


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _train_epoch(
    model: CBMFunnyBirds,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_c: float,
    freeze_backbone: bool,
) -> Tuple[float, float, float]:
    if freeze_backbone:
        model.backbone.eval()
        model.concept_head.train()
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
            c_logits = model.concept_head(feats)
            y_logits = model.label_head(torch.sigmoid(c_logits))
        else:
            y_logits, c_logits = model(imgs)

        task_loss = ce_fn(y_logits, y)
        c_loss    = bce_fn(c_logits, c)
        loss      = task_loss + lambda_c * c_loss

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
    model: CBMFunnyBirds,
    loader: DataLoader,
    device: torch.device,
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

        y_logits, c_logits = model(imgs)

        task_loss  = ce_fn(y_logits, y)
        c_loss     = bce_fn(c_logits, c)
        total_sum += (task_loss + lambda_c * c_loss).item()

        c_preds    = (torch.sigmoid(c_logits) > 0.5).float()
        c_correct += (c_preds == c).sum().item()
        c_total   += c.numel()

        t_correct += (y_logits.argmax(dim=1) == y).sum().item()
        t_total   += y.size(0)

    n = len(loader)
    return total_sum / n, c_correct / c_total, t_correct / t_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CBM on FunnyBirds (50 classes, 26 concepts)"
    )
    parser.add_argument("--funnybirds_root", type=str, required=True)
    parser.add_argument("--checkpoint_dir",  type=str,
                        default="checkpoints_funnybirds")
    parser.add_argument("--backbone_ckpt",   type=str,
                        default="checkpoints_funnybirds/resnet50_funnybirds_best.pth")
    parser.add_argument("--epochs_stage1",   type=int,   default=12)
    parser.add_argument("--epochs_stage2",   type=int,   default=10)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lambda_c",        type=float, default=1.0)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--device",          type=str,   default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[CBM-FB] Using device: {device}")
    print(f"[CBM-FB] lambda_c={args.lambda_c}")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, n_classes = build_loaders(
        args.funnybirds_root, args.batch_size, args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CBMFunnyBirds(num_concepts=NUM_CONCEPTS, num_classes=n_classes)

    # Initialise backbone from pre-trained FunnyBirds ResNet-50
    state = torch.load(args.backbone_ckpt, map_location="cpu", weights_only=True)
    backbone_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"[CBM-FB] Backbone init from {args.backbone_ckpt}")
    if missing:
        print(f"[CBM-FB] missing keys: {missing}")
    if unexpected:
        print(f"[CBM-FB] unexpected keys: {unexpected}")

    model.to(device)

    # ── Stage 1: backbone frozen ───────────────────────────────────────────────
    print(f"\n[CBM-FB] Stage 1: backbone frozen, {args.epochs_stage1} epochs")
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt1 = torch.optim.AdamW(
        list(model.concept_head.parameters()) +
        list(model.label_head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    best_stage1_c_acc = 0.0
    for epoch in range(1, args.epochs_stage1 + 1):
        tr_loss, tr_c_loss, tr_task = _train_epoch(
            model, train_loader, opt1, device,
            lambda_c=args.lambda_c, freeze_backbone=True,
        )
        val_loss, val_c_acc, val_task_acc = _eval_epoch(
            model, val_loader, device, lambda_c=args.lambda_c,
        )
        print(
            f"[Concepts][{epoch:2d}]  "
            f"tr_loss={tr_loss:.4f}  tr_c_loss={tr_c_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_c_acc={val_c_acc:.4f}  "
            f"val_task_acc={val_task_acc:.4f}"
        )
        best_stage1_c_acc = max(best_stage1_c_acc, val_c_acc)

    # ── Stage 2: full fine-tune ────────────────────────────────────────────────
    print(f"\n[CBM-FB] Stage 2: full fine-tune, {args.epochs_stage2} epochs")
    for p in model.backbone.parameters():
        p.requires_grad = True

    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs_stage2 + 1):
        tr_loss, tr_c_loss, tr_task = _train_epoch(
            model, train_loader, opt2, device,
            lambda_c=args.lambda_c, freeze_backbone=False,
        )
        val_loss, val_c_acc, val_task_acc = _eval_epoch(
            model, val_loader, device, lambda_c=args.lambda_c,
        )
        print(
            f"[Labels][{epoch:2d}]  "
            f"tr_loss={tr_loss:.4f}  tr_task={tr_task:.4f}  "
            f"val_loss={val_loss:.4f}  val_c_acc={val_c_acc:.4f}  "
            f"val_task_acc={val_task_acc:.4f}"
        )

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = out_dir / "cbm_funnybirds.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "num_concepts":  NUM_CONCEPTS,
                "num_classes":   n_classes,
                "lambda_c":      args.lambda_c,
                "concept_names": concept_names(),
                "dataset":       "funnybirds",
            },
            "best_stage1_val_c_acc": best_stage1_c_acc,
        },
        ckpt_path,
    )
    print(f"\n[CBM-FB] Saved checkpoint to {ckpt_path}")
    print(f"[CBM-FB] num_concepts={NUM_CONCEPTS}  best_stage1_val_c_acc={best_stage1_c_acc:.4f}")


if __name__ == "__main__":
    main()
