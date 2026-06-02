"""
datasets/funnybirds_dataset.py

PyTorch Dataset for the FunnyBirds synthetic bird dataset.
https://github.com/visinf/funnybirds  (ICCV 2023)

Expected directory structure
-----------------------------
<funnybirds_root>/
    data/
        0_0000.png          # images: {class_id}_{index:04d}.png
        0_0001.png
        ...
    annotation.json         # list of per-image dicts (see below)
    classes.json            # class_id -> part combination dict

annotation.json entry:
    {
        "id":        0,
        "file_name": "0_0000.png",
        "class":     0,
        "beak":      2,   # variant index (0-based)
        "eye":       0,
        "wing":      3,
        "foot":      1,
        "tail":      5,
        "bg":        4,
        "split":     "train"   # "train" or "test"
    }

classes.json:
    {
        "0": {"beak": 2, "eye": 0, "wing": 3, "foot": 1, "tail": 5},
        "1": {"beak": 0, "eye": 2, "wing": 1, "foot": 3, "tail": 7},
        ...
    }

Concepts (26 binary one-hot dimensions, perfectly balanced by construction):
    beak_0 .. beak_3   (4 dims)
    eye_0  .. eye_2    (3 dims)
    wing_0 .. wing_5   (6 dims)
    foot_0 .. foot_3   (4 dims)
    tail_0 .. tail_8   (9 dims)

These are constructed from annotation["beak"], ["eye"], etc.
No 10-90% prevalence filter needed — each variant is used by exactly
50 / num_variants species (perfectly balanced by construction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Part metadata (number of variants per part; FunnyBirds spec)
# ---------------------------------------------------------------------------

PART_VARIANTS: Dict[str, int] = {
    "beak": 4,
    "eye":  3,
    "wing": 6,
    "foot": 4,
    "tail": 9,
}
PARTS: List[str] = list(PART_VARIANTS.keys())
NUM_CONCEPTS: int = sum(PART_VARIANTS.values())  # 26


def concept_names() -> List[str]:
    """Return the 26 binary concept names in canonical order."""
    names = []
    for part, n_var in PART_VARIANTS.items():
        for v in range(n_var):
            names.append(f"{part}_{v}")
    return names


def parts_to_concept_vector(annotation: Dict[str, Any]) -> torch.Tensor:
    """
    Convert an annotation dict (with 'beak', 'eye', 'wing', 'foot', 'tail' keys)
    to a 26-dim one-hot concept tensor.

    For example, if annotation['beak'] == 2 and PART_VARIANTS['beak'] == 4,
    then dims for beak_0..beak_3 = [0, 0, 1, 0].
    """
    vec = []
    for part, n_var in PART_VARIANTS.items():
        v = int(annotation[part])
        one_hot = [0.0] * n_var
        one_hot[v] = 1.0
        vec.extend(one_hot)
    return torch.tensor(vec, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FunnyBirdsDataset(Dataset):
    """
    PyTorch Dataset for FunnyBirds.

    Returns a dict with:
        image       : transformed image tensor (C, H, W)
        label       : species class id, 0-based (long)
        image_id    : unique integer per image
        concepts    : 26-dim float32 one-hot concept vector (if include_concepts=True)
    """

    def __init__(
        self,
        funnybirds_root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        include_concepts: bool = False,
    ):
        super().__init__()
        self.root = Path(funnybirds_root)
        self.split = split
        self.transform = transform
        self.include_concepts = include_concepts

        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"

        ann_path = self.root / "annotation.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"annotation.json not found at {ann_path}. "
                "Run scripts/prepare_funnybirds_metadata.py or "
                "download FunnyBirds from https://github.com/visinf/funnybirds"
            )

        with open(ann_path) as f:
            all_ann = json.load(f)

        self.samples: List[Dict[str, Any]] = [
            a for a in all_ann if a.get("split", "train") == split
        ]

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split={split!r} in {ann_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.samples[idx]

        # Image
        img_path = self.root / "data" / ann["file_name"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Label: class index 0-based (annotation stores 0-based class_id)
        label = int(ann["class"])

        # image_id: use "id" field if present; fall back to idx
        image_id = int(ann.get("id", idx))

        sample: Dict[str, Any] = {
            "image":    img,
            "label":    torch.tensor(label,    dtype=torch.long),
            "image_id": image_id,
        }

        if self.include_concepts:
            sample["concepts"] = parts_to_concept_vector(ann)

        return sample

    def num_classes(self) -> int:
        """Number of unique species classes."""
        return len({a["class"] for a in self.samples})

    def get_class_concept_matrix(self) -> Tuple[torch.Tensor, List[str]]:
        """
        Returns (matrix, concept_names) where matrix has shape (num_classes, 26).
        Row i is the concept vector for class i.
        All images of class i share the same concept vector (by construction).
        """
        num_cls = self.num_classes()
        matrix = torch.zeros(num_cls, NUM_CONCEPTS)
        seen = set()
        for ann in self.samples:
            c = int(ann["class"])
            if c not in seen:
                matrix[c] = parts_to_concept_vector(ann)
                seen.add(c)
        return matrix, concept_names()
