# src/datasets/cub_dataset.py

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from .cub_metadata import load_cub_metadata, CUBMetadata


class CUBDataset(Dataset):
    """
    PyTorch Dataset for CUB_200_2011, using metadata CSVs.

    Returns a dict with:
      - image: transformed image tensor
      - label: species class id (0-based)
      - image_id: integer
      - part_concepts: binary vector of part presence
    """

    def __init__(
        self,
        cub_root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        include_part_concepts: bool = False,
    ):
        super().__init__()
        self.cub_root = Path(cub_root)
        self.meta: CUBMetadata = load_cub_metadata(self.cub_root)
        self.transform = transform
        self.include_part_concepts = include_part_concepts

        assert split in ("train", "test")
        is_train = 1 if split == "train" else 0

        images = self.meta.images
        self.images_df = images[images["is_train"] == is_train].copy()
        self.images_df = self.images_df.reset_index(drop=True)

        # Prepare part_concepts lookup if requested
        self.part_concepts_map: Optional[Dict[int, torch.Tensor]] = None
        self.part_cols = None

        if include_part_concepts and self.meta.image_parts_binary is not None:
            ipb = self.meta.image_parts_binary.set_index("image_id")
            part_cols = [c for c in ipb.columns if c.startswith("part_")]
            self.part_cols = part_cols

            self.part_concepts_map = {}
            for img_id, row in ipb.iterrows():
                vec = torch.tensor(row[part_cols].values, dtype=torch.float32)
                self.part_concepts_map[int(img_id)] = vec

    def __len__(self) -> int:
        return len(self.images_df)

    def _resolve_image_path(self, rel_path: str) -> Path:
        """
        CUB file_path entries typically look like '001.Black_footed_Albatross/Black_footed_Albatross_0001_796111.jpg'
        or sometimes 'images/001.Black_footed_Albatross/...'.
        We always want cv_root/images/<class_dir>/<file>.
        """
        if rel_path.startswith("images/"):
            rel_path = rel_path.split("images/", 1)[1]
        return self.cub_root / "images" / rel_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.images_df.iloc[idx]
        img_path = self._resolve_image_path(row["file_path"])
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # CUB class_id is 1..200; we map to 0..199
        label = int(row["class_id"]) - 1
        image_id = int(row["image_id"])

        sample: Dict[str, Any] = {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": image_id,
        }

        if self.include_part_concepts and self.part_concepts_map is not None:
            sample["part_concepts"] = self.part_concepts_map.get(image_id, None)

        return sample
