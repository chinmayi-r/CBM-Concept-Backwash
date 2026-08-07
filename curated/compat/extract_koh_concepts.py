#!/usr/bin/env python3
"""Batched compatibility wrapper for Koh ``ExtractConcepts``.

It preserves the upstream train/eval transforms and sigmoid convention while
loading old full-model checkpoints explicitly on modern PyTorch.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def image_path(record, work_dir: Path) -> Path:
    raw = str(record["img_path"])
    pieces = raw.replace("\\", "/").split("/")
    if "CUB_200_2011" in pieces:
        return work_dir / Path(*pieces[pieces.index("CUB_200_2011"):])
    return Path(raw)


class Images(Dataset):
    def __init__(self, records, work_dir, transform):
        self.records, self.work_dir, self.transform = records, work_dir, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        image = Image.open(image_path(self.records[index], self.work_dir)).convert("RGB")
        return self.transform(image)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--koh-root", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    sys.path.insert(0, str(args.koh_root.resolve()))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(args.model, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(args.model, map_location=device)
    model = model.to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (args.data_dir / f"{split}.pkl").open("rb") as stream:
            records = pickle.load(stream)
        if split == "train":
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(299), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[2.0] * 3),
            ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(299), transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[2.0] * 3),
            ])
        loader = DataLoader(Images(records, args.work_dir, transform), batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)
        values = []
        with torch.inference_mode():
            for images in loader:
                outputs = model(images.to(device))
                logits = torch.cat([item.reshape(item.shape[0], -1) for item in outputs], dim=1)
                values.extend(torch.sigmoid(logits).cpu().tolist())
        if len(values) != len(records):
            raise RuntimeError(f"{split}: extracted {len(values)} rows for {len(records)} records")
        updated = []
        for record, concepts in zip(records, values):
            row = copy.deepcopy(record); row["attribute_label"] = concepts; updated.append(row)
        with (args.out_dir / f"{split}.pkl").open("wb") as stream:
            pickle.dump(updated, stream)
        print(f"[KOH EXTRACT SUCCESS] {split}: {len(updated)}")


if __name__ == "__main__":
    main()
