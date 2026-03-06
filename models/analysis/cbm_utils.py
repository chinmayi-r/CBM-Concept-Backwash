# analysis/cbm_utils.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from datasets.cub_metadata import load_cub_metadata
from models.cbm_cub import CBMConfig, SimpleCBM


@dataclass
class CUBConceptsConfig:
    cub_root: str | Path
    split: Literal["train", "test"]
    part_cols: Sequence[str]


class CUBConceptsDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset for CBM:
      returns (image_path, species_label, concept_vector)
    """

    def __init__(self, cfg: CUBConceptsConfig):
        self.cub_root = Path(cfg.cub_root)
        self.cfg = cfg

        meta = load_cub_metadata(self.cub_root)
        images = meta.images.set_index("image_id")
        labels = meta.image_class_labels.set_index("image_id")
        split = meta.train_test_split.set_index("image_id")
        ipb = meta.image_parts_binary.set_index("image_id")

        if cfg.split == "train":
            mask = split["is_training_image"] == 1
        else:
            mask = split["is_training_image"] == 0

        self.image_ids = split.index[mask].tolist()
        self.images_df = images
        self.labels_df = labels
        self.ipb_df = ipb
        self.part_cols = list(cfg.part_cols)

        species_labels = []
        concept_vectors = []
        for img_id in self.image_ids:
            cls_id = int(labels.loc[img_id, "class_id"]) - 1
            species_labels.append(cls_id)

            if img_id in ipb.index:
                row = ipb.loc[img_id]
                concepts = [int(row.get(col, 0)) for col in self.part_cols]
            else:
                concepts = [0] * len(self.part_cols)

            concept_vectors.append(concepts)

        self.species_labels = torch.tensor(species_labels, dtype=torch.long)
        self.concept_vectors = torch.tensor(concept_vectors, dtype=torch.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        rel_path = self.images_df.loc[img_id, "file_path"]
        img_path = self.cub_root / "images" / rel_path
        y_species = self.species_labels[idx]
        concepts = self.concept_vectors[idx]
        return img_path, y_species, concepts


class ImageLoaderWrapper(torch.utils.data.Dataset):
    """
    Wraps CUBConceptsDataset that returns paths
    into a dataset that returns actual images.
    """

    def __init__(self, base_ds: CUBConceptsDataset, transform):
        self.base_ds = base_ds
        self.transform = transform
        self.Image = Image

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img_path, y_species, concepts = self.base_ds[idx]
        img = self.Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, y_species, concepts


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


def train_epoch_concepts(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0
    n_correct = 0
    n_total_labels = 0

    for x, _, c in tqdm(loader, desc="Train concepts", leave=False):
        x = x.to(device)
        c = c.to(device)

        optimizer.zero_grad()
        _, concept_logits = model(x)
        loss = criterion(concept_logits, c)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

        preds = (concept_logits.sigmoid() > 0.5).float()
        n_correct += (preds == c).sum().item()
        n_total_labels += c.numel()

    avg_loss = total_loss / n_samples
    acc = n_correct / n_total_labels
    return avg_loss, acc


def eval_concepts(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0
    n_correct = 0
    n_total_labels = 0

    with torch.no_grad():
        for x, _, c in tqdm(loader, desc="Eval concepts", leave=False):
            x = x.to(device)
            c = c.to(device)
            _, concept_logits = model(x)
            loss = criterion(concept_logits, c)

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            preds = (concept_logits.sigmoid() > 0.5).float()
            n_correct += (preds == c).sum().item()
            n_total_labels += c.numel()

    return total_loss / n_samples, n_correct / n_total_labels


def train_epoch_labels(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_samples = 0
    n_correct = 0

    for x, y, _ in tqdm(loader, desc="Train labels", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        label_logits, _ = model(x)
        loss = criterion(label_logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = label_logits.argmax(dim=1)
        n_correct += (preds == y).sum().item()
        n_samples += x.size(0)

    return total_loss / n_samples, n_correct / n_samples


def eval_labels(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_samples = 0
    n_correct = 0

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Eval labels", leave=False):
            x = x.to(device)
            y = y.to(device)
            label_logits, _ = model(x)
            loss = criterion(label_logits, y)

            total_loss += loss.item() * x.size(0)
            preds = label_logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_samples += x.size(0)

    return total_loss / n_samples, n_correct / n_samples


def build_cbm_and_loaders(
    cub_root: str | Path,
    part_cols: Sequence[str],
    batch_size: int = 64,
    backbone: str = "resnet18",
    num_workers: int = 4,
):
    cub_root = Path(cub_root)

    tf_train, tf_eval = make_transforms()

    cfg_train = CUBConceptsConfig(cub_root=cub_root, split="train", part_cols=part_cols)
    cfg_test = CUBConceptsConfig(cub_root=cub_root, split="test", part_cols=part_cols)

    ds_train_base = CUBConceptsDataset(cfg_train)
    ds_test_base = CUBConceptsDataset(cfg_test)

    ds_train = ImageLoaderWrapper(ds_train_base, tf_train)
    ds_test = ImageLoaderWrapper(ds_test_base, tf_eval)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    cbm_cfg = CBMConfig(
        num_concepts=len(part_cols),
        num_classes=200,
        backbone_name=backbone,
        pretrained=True,
        freeze_backbone=False,
    )
    model = SimpleCBM(cbm_cfg)

    return model, train_loader, test_loader
