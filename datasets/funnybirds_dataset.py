"""
datasets/funnybirds_dataset.py

PyTorch Dataset for the FunnyBirds synthetic bird dataset.
https://github.com/visinf/funnybirds  (ICCV 2023)

Dataset download (NOT the code repo — that's separate):
    wget https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
    unzip FunnyBirds.zip
    mv FunnyBirds data/FunnyBirds

Expected directory structure after unzipping
---------------------------------------------
<funnybirds_root>/
    dataset_train.json      # JSON array, one dict per training image
    dataset_test.json       # JSON array, one dict per test image
    parts.json              # dict: {part_name: [variant_dict, ...]}
    classes.json            # dict: {class_idx: {part: variant_dict, ...}}
    train/
        0/
            000000.png      # image idx zero-padded to 6 digits
            000001.png
            ...
        1/ ...
        49/ ...
    test/
        0/ ...
        ...

Annotation entry (one element of dataset_train.json / dataset_test.json):
    {
        "class_idx":  0,
        "beak_model": "beak01.glb",
        "eye_model":  "eye02.glb",
        "foot_model": "foot03.glb",
        "tail_model": "tail01.glb",  "tail_color": "red",
        "wing_model": "wing02.glb",  "wing_color": "green",
        ... (camera, lighting, background fields — ignored)
    }

parts.json structure:
    {
        "beak": [{"model": "beak01.glb"}, {"model": "beak02.glb"}, ...],
        "eye":  [{"model": "eye01.glb"}, ...],
        "foot": [{"model": "foot01.glb"}, ...],
        "tail": [{"model": "tail01.glb", "color": "red"}, ...],
        "wing": [{"model": "wing01.glb", "color": "red"}, ...]
    }
    Variant index = position in the list.
    beak/eye/foot: only model matters.
    tail/wing: both model AND color matter.

Concepts (26 binary one-hot dimensions, perfectly balanced by construction):
    beak_0 .. beak_3   (4 dims)
    eye_0  .. eye_2    (3 dims)
    wing_0 .. wing_5   (6 dims)
    foot_0 .. foot_3   (4 dims)
    tail_0 .. tail_8   (9 dims)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Part metadata
# ---------------------------------------------------------------------------

PARTS: List[str] = ["beak", "eye", "wing", "foot", "tail"]
PART_VARIANTS: Dict[str, int] = {"beak": 4, "eye": 3, "wing": 6, "foot": 4, "tail": 9}
NUM_CONCEPTS: int = sum(PART_VARIANTS.values())  # 26
_FUNNYBIRDS_N_TRAIN: int = 50_000  # 50 classes × 1000 images/class; used for global image IDs


def concept_names() -> List[str]:
    """Return the 26 binary concept names in canonical order."""
    names = []
    for part, n_var in PART_VARIANTS.items():
        for v in range(n_var):
            names.append(f"{part}_{v}")
    return names


# ---------------------------------------------------------------------------
# Part lookup: (model [+ color]) → variant index
# ---------------------------------------------------------------------------

def _build_part_lookup(parts_json: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Build a lookup table from part params → variant index.

    parts_json: loaded parts.json  {part_name: [variant_dict, ...]}
    Returns: {part_name: {(sorted key-value pairs) -> int}}
    """
    lookup: Dict[str, Dict] = {}
    for part, variants in parts_json.items():
        lookup[part] = {}
        for idx, vd in enumerate(variants):
            key_fields: Dict[str, str] = {"model": vd["model"]}
            if "color" in vd:
                key_fields["color"] = vd["color"]
            key = tuple(sorted(key_fields.items()))
            lookup[part][key] = idx
    return lookup


def _params_to_variant_idx(
    lookup: Dict[str, Dict], part: str, entry: Dict[str, Any]
) -> int:
    """Convert a raw annotation entry to the variant index for one part."""
    model = entry[f"{part}_model"]
    key_fields: Dict[str, str] = {"model": model}
    color_key = f"{part}_color"
    if color_key in entry:
        key_fields["color"] = entry[color_key]
    key = tuple(sorted(key_fields.items()))
    return lookup[part][key]


def params_to_concept_vector(
    lookup: Dict[str, Dict], entry: Dict[str, Any]
) -> torch.Tensor:
    """
    Convert an annotation entry to a 26-dim one-hot concept vector.
    Uses the parts.json lookup for correct model+color → index mapping.
    """
    vec: List[float] = []
    for part, n_var in PART_VARIANTS.items():
        v = _params_to_variant_idx(lookup, part, entry)
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
        image_id    : unique integer per image (= array index in dataset_{split}.json)
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

        ann_path = self.root / f"dataset_{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"{ann_path} not found.\n"
                "Download the FunnyBirds dataset (not the code repo):\n"
                "  wget https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip\n"
                "  unzip FunnyBirds.zip && mv FunnyBirds <funnybirds_root>"
            )

        with open(ann_path) as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        # Load parts.json for concept vector construction
        parts_path = self.root / "parts.json"
        if not parts_path.exists():
            raise FileNotFoundError(f"parts.json not found at {parts_path}")
        with open(parts_path) as f:
            parts_json = json.load(f)
        self.lookup = _build_part_lookup(parts_json)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.samples[idx]
        class_idx = int(entry["class_idx"])

        # Image path: {root}/{mode}/{class_idx}/{idx:06d}.png
        img_path = self.root / self.split / str(class_idx) / f"{idx:06d}.png"
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Global image_id: train=idx, test=_FUNNYBIRDS_N_TRAIN+idx (avoids overlap)
        global_id = idx if self.split == "train" else _FUNNYBIRDS_N_TRAIN + idx
        sample: Dict[str, Any] = {
            "image":    img,
            "label":    torch.tensor(class_idx, dtype=torch.long),
            "image_id": global_id,
        }

        if self.include_concepts:
            sample["concepts"] = params_to_concept_vector(self.lookup, entry)

        return sample

    def num_classes(self) -> int:
        return len({int(e["class_idx"]) for e in self.samples})

    def get_class_concept_matrix(self) -> Tuple[torch.Tensor, List[str]]:
        """
        Returns (matrix, concept_names) where matrix[i] is the concept vector for class i.
        All images of class i share the same concept vector (by construction).
        """
        num_cls = self.num_classes()
        matrix = torch.zeros(num_cls, NUM_CONCEPTS)
        seen: set = set()
        for entry in self.samples:
            c = int(entry["class_idx"])
            if c not in seen:
                matrix[c] = params_to_concept_vector(self.lookup, entry)
                seen.add(c)
        return matrix, concept_names()
