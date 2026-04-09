#!/usr/bin/env python
"""
scripts/train_resnet_funnybirds.py

Pre-train a ResNet-50 backbone on FunnyBirds (50-class species classification).

Mirrors scripts/train_resnet.py exactly:
  - Same architecture (ResNet-50, fc replaced with Linear(2048, num_classes))
  - Same optimizer (AdamW)
  - Same transforms (ImageNet normalization)
  - Same checkpoint format
  - Same train/eval loop structure

The only differences vs train_resnet.py:
  - Uses FunnyBirdsDataset instead of CUBDataset
  - num_classes = 50 (not 200)
  - Default --out_ckpt points to checkpoints_funnybirds/

Usage:
    python -m scripts.train_resnet_funnybirds \\
        --funnybirds_root data/FunnyBirds \\
        --epochs 50 \\
        --batch_size 64 \\
        --out_ckpt checkpoints_funnybirds/resnet50_funnybirds_best.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from datasets.funnybirds_dataset import FunnyBirdsDataset


# ---------------------------------------------------------------------------
# Transforms (mirror train_resnet.py exactly)
# ---------------------------------------------------------------------------

def make_transforms():
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


def get_loaders(funnybirds_root: str, batch_size: int = 64, num_workers: int = 4):
    tf_train, tf_eval = make_transforms()
    train_ds = FunnyBirdsDataset(funnybirds_root, split="train", transform=tf_train)
    test_ds  = FunnyBirdsDataset(funnybirds_root, split="test",  transform=tf_eval)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    n_classes = train_ds.num_classes()
    print(f"[train_resnet_fb] FunnyBirds: {len(train_ds)} train, {len(test_ds)} test, {n_classes} classes")
    return train_loader, test_loader, n_classes


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total   = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="Train" if train else "Eval", leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train ResNet-50 on FunnyBirds (50-class)"
    )
    parser.add_argument("--funnybirds_root", type=str, required=True)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--out_ckpt",        type=str,
                        default="checkpoints_funnybirds/resnet50_funnybirds_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_resnet_fb] Using device:", device)

    train_loader, test_loader, n_classes = get_loaders(
        args.funnybirds_root, args.batch_size, args.num_workers
    )

    # ResNet-50 with ImageNet weights, replace fc for FunnyBirds num_classes
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"[train_resnet_fb] Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, test_loader,  criterion, optimizer, device, train=False)
        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_ckpt)
            print(f"  -> New best val_acc={best_val_acc:.3f}; saved to {out_ckpt}")

    print("[train_resnet_fb] Finished. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
