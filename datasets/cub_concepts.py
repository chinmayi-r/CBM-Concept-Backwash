# datasets/cub_concepts.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.cub_metadata import load_cub_metadata


@dataclass
class CUBConceptsConfig:
    cub_root: str | Path
    split: str  # "train" or "test"
    part_cols: Sequence[str]  # e.g. ["part_1_present", ..., "part_10_present"]


class CUBConceptsDataset(Dataset):
    """
    Returns (image_path, species_label, concept_vector) for each image.
    This is a lightweight dataset – we just expose paths and labels.
    You can wrap it with a transform-based dataset if needed.
    """

    def __init__(self, cfg: CUBConceptsConfig):
        self.cub_root = Path(cfg.cub_root)
        self.cfg = cfg

        meta = load_cub_metadata(self.cub_root)
        self.images_df = meta.images.set_index("image_id")
        self.labels_df = meta.image_class_labels.set_index("image_id")
        self.split_df = meta.train_test_split.set_index("image_id")
        self.ipb_df = meta.image_parts_binary.set_index("image_id")

        if cfg.split == "train":
            mask = self.split_df["is_training_image"] == 1
        elif cfg.split == "test":
            mask = self.split_df["is_training_image"] == 0
        else:
            raise ValueError(f"Unknown split: {cfg.split}")

        self.image_ids = self.split_df.index[mask].tolist()
        self.part_cols = list(cfg.part_cols)

        # Precompute labels and concepts as tensors for speed
        species_labels = []
        concept_vectors = []

        for img_id in self.image_ids:
            cls_id = int(self.labels_df.loc[img_id, "class_id"]) - 1  # 0-based
            species_labels.append(cls_id)

            if img_id in self.ipb_df.index:
                row = self.ipb_df.loc[img_id]
                concepts = [int(row.get(col, 0)) for col in self.part_cols]
            else:
                concepts = [0] * len(self.part_cols)

            concept_vectors.append(concepts)

        self.species_labels = torch.tensor(species_labels, dtype=torch.long)
        self.concept_vectors = torch.tensor(concept_vectors, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        rel_path = self.images_df.loc[img_id, "file_path"]
        img_path = self.cub_root / "images" / rel_path

        y_species = self.species_labels[idx]
        concepts = self.concept_vectors[idx]

        return img_path, y_species, concepts
