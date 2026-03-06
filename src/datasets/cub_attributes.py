cub_attributes# src/datasets/cub_attributes.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

# NOTE: cub_metadata lives in top-level "datasets", not under "src"
from datasets.cub_metadata import load_cub_metadata


@dataclass
class CUBAttributesConfig:
    cub_root: str | Path
    split: str              # "train" or "test"
    attr_cols: Sequence[str]  # e.g. ["attr_10_present", "attr_155_present"]


class CUBAttributesDataset(Dataset):
    """
    Returns (image_path, species_label, attribute_vector) for each image.

    - image_path: Path into cub_root/images
    - species_label: int in [0, 199]
    - attribute_vector: float tensor of shape [K] with 0/1 entries
    """

    def __init__(self, cfg: CUBAttributesConfig):
        self.cub_root = Path(cfg.cub_root)
        self.cfg = cfg

        meta = load_cub_metadata(self.cub_root)

        images_df = meta.images.set_index("image_id")
        labels_df = meta.image_class_labels.set_index("image_id")
        split_df = meta.train_test_split.set_index("image_id")

        if meta.image_attributes_binary is None:
            raise RuntimeError(
                "image_attributes_binary is None – did you parse attributes correctly?"
            )
        attr_df = meta.image_attributes_binary.set_index("image_id")

        if cfg.split == "train":
            mask = split_df["is_training_image"] == 1
        elif cfg.split == "test":
            mask = split_df["is_training_image"] == 0
        else:
            raise ValueError(f"Unknown split: {cfg.split}")

        self.image_ids = split_df.index[mask].tolist()
        self.images_df = images_df
        self.labels_df = labels_df
        self.attr_df = attr_df
        self.attr_cols = list(cfg.attr_cols)

        species_labels = []
        attr_vectors = []

        for img_id in self.image_ids:
            cls_id = int(self.labels_df.loc[img_id, "class_id"]) - 1  # 0-based
            species_labels.append(cls_id)

            if img_id in self.attr_df.index:
                row = self.attr_df.loc[img_id]
                vec = [int(row.get(col, 0)) for col in self.attr_cols]
            else:
                vec = [0] * len(self.attr_cols)

            attr_vectors.append(vec)

        self.species_labels = torch.tensor(species_labels, dtype=torch.long)
        self.attr_vectors = torch.tensor(attr_vectors, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        rel_path = self.images_df.loc[img_id, "file_path"]
        img_path = self.cub_root / "images" / rel_path

        y_species = self.species_labels[idx]
        attrs = self.attr_vectors[idx]

        return img_path, y_species, attrs
