#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.cub_dataset import CUBDataset
from models.resnet_cub import get_resnet50_cub, save_checkpoint


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


def get_loaders(cub_root: str, batch_size: int = 64, num_workers: int = 4):
    tf_train, tf_eval = make_transforms()
    train_ds = CUBDataset(cub_root, split="train", transform=tf_train)
    test_ds = CUBDataset(cub_root, split="test", transform=tf_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="Train" if train else "Eval", leave=False):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_ckpt", type=str, default="checkpoints/resnet50_cub_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_resnet] Using device:", device)

    train_loader, test_loader = get_loaders(args.cub_root, args.batch_size, args.num_workers)

    model = get_resnet50_cub().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"[train_resnet] Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

        print(
            f"  train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, str(out_ckpt))
            print(f"  -> New best val_acc={best_val_acc:.3f}; saved to {out_ckpt}")

    print("[train_resnet] Finished. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
