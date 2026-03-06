# scripts/train_cbm.py

from __future__ import annotations
import argparse
from pathlib import Path
import torch

from analysis.cbm_utils import (
    build_cbm_and_loaders,
    train_epoch_concepts,
    eval_concepts,
    train_epoch_labels,
    eval_labels,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints_cbm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_concepts", type=int, default=10)
    parser.add_argument("--epochs_labels", type=int, default=10)
    parser.add_argument("--lr_concepts", type=float, default=1e-3)
    parser.add_argument("--lr_labels", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen_part_cols = [f"part_{i}_present" for i in range(1, 11)]
    model, train_loader, test_loader = build_cbm_and_loaders(
        args.cub_root, chosen_part_cols, batch_size=args.batch_size, backbone=args.backbone,
    )
    model.to(device)

    # Stage 1: concepts
    opt_concepts = torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.concept_head.parameters()),
        lr=args.lr_concepts,
    )
    for epoch in range(1, args.epochs_concepts + 1):
        train_loss, train_acc = train_epoch_concepts(model, train_loader, opt_concepts, device)
        val_loss, val_acc = eval_concepts(model, test_loader, device)
        print(f"[Concepts][{epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    torch.save(model.state_dict(), out_dir / "cbm_concepts_only.pt")

    # Stage 2: labels
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.concept_head.parameters():
        p.requires_grad = False

    opt_labels = torch.optim.Adam(model.label_head.parameters(), lr=args.lr_labels)
    for epoch in range(1, args.epochs_labels + 1):
        train_loss, train_acc = train_epoch_labels(model, train_loader, opt_labels, device)
        val_loss, val_acc = eval_labels(model, test_loader, device)
        print(f"[Labels][{epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    torch.save(model.state_dict(), out_dir / "cbm_full.pt")


if __name__ == "__main__":
    main()
