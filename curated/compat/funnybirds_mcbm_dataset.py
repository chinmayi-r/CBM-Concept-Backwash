"""
curated/compat/funnybirds_mcbm_dataset.py

FunnyBirds dataset adapter for the official minimal_cbm trainer
(antonioalmudevar/minimal_cbm). The upstream repo has no FunnyBirds support
(src/datasets/__init__.py's get_loader() only dispatches CUB200, CIFAR10,
DSPRITES/MPI3D/SHAPES3D, CELEBA, SPIRALS) -- so this is new code, not a port
of anything upstream. It mirrors src/datasets/cub200.py's interface exactly
(get_cub200 / CUB200 / find_class_imbalance) so it's a drop-in replacement
when registered via the get_loader monkey-patch in
curated/patches/run_mcbm_funnybirds.py. external/ itself is never edited.

Reads the same train.pkl/test.pkl produced by
curated/data/funnybirds/build_funnybirds_cbm_data.py (id, img_path,
class_label, attribute_label, attribute_certainty) -- the identical schema
the official CBM trainer consumes, so both frameworks train from one data
build.

Concept groups: FunnyBirds concepts are one-hot per part (beak/eye/wing/foot/
tail); the schema (which indices belong to which part) is derived at runtime
from the OFFICIAL parts.json via curated/data/funnybirds/funnybirds_concepts.py
(load_parts/group_slices), not a hardcoded constant -- group_slices(parts)
returns an OrderedDict[part_name, (start, end)], converted below into the
list-of-index-lists format minimal_cbm's concepts_groups expects.
"""

from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_FB_DATA = _REPO / "curated" / "data" / "funnybirds"
if str(_FB_DATA) not in sys.path:
    sys.path.insert(0, str(_FB_DATA))

from funnybirds_concepts import load_parts, group_slices  # noqa: E402


def find_class_imbalance(data: list[dict], multiple_attr: bool = True) -> list[float]:
    """Verbatim port of src/datasets/cub200.find_class_imbalance (multiple_attr=True path)."""
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    n_ones = [0] * n_attr
    total = [n] * n_attr
    for d in data:
        labels = d["attribute_label"]
        for i in range(n_attr):
            n_ones[i] += labels[i]
    return [total[j] / n_ones[j] - 1 if n_ones[j] > 0 else 0.0 for j in range(n_attr)]


class FunnyBirdsMCBM(Dataset):
    """Mirrors src/datasets/cub200.CUB200's public interface."""

    def __init__(
        self,
        train: bool,
        pkls_dir: str,
        funnybirds_root: str,
        n_classes: int = 50,
        transform=None,
        return_nuisances: bool = False,
        **kwargs,
    ) -> None:
        self.train = train
        self.pkls_dir = pkls_dir
        self.transform = transform
        self.n_classes = n_classes
        self.return_nuisances = return_nuisances

        pkl_file = os.path.join(pkls_dir, "train.pkl" if train else "test.pkl")
        with open(pkl_file, "rb") as f:
            self.data: list[dict] = pickle.load(f)

        parts = load_parts(funnybirds_root)
        self.concepts_groups = group_slices_to_index_lists(parts)
        self.n_concepts = sum(len(g) for g in self.concepts_groups)
        assert self.n_concepts == len(self.data[0]["attribute_label"]), (
            f"parts.json implies {self.n_concepts} concepts but pkl records have "
            f"{len(self.data[0]['attribute_label'])} -- rebuild with the same "
            f"funnybirds_root used by build_funnybirds_cbm_data.py"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        entry = self.data[index]
        image = Image.open(entry["img_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        task = entry["class_label"]
        concepts = torch.tensor(entry["attribute_label"], dtype=torch.float32)
        if self.return_nuisances:
            # FunnyBirds' 26 concepts are already the full bottleneck (no
            # held-out "nuisance" attributes), so both nuisance tensors are
            # empty -- mirrors cub200.CUB200.__getitem__'s return_nuisances
            # shape (image, task, concepts, nuisances_task, nuisances_nontask)
            # which src/experiments/train.py unpacks unconditionally
            # (`for x, y, c, _, _ in self.train_loader`).
            empty = torch.tensor([]).float()
            return image, task, concepts, empty, empty
        return image, task, concepts


def group_slices_to_index_lists(parts) -> list[list[int]]:
    """Convert funnybirds_concepts.group_slices(parts) (OrderedDict[part, (start,
    stop)]) to the list-of-index-lists format minimal_cbm's concepts_groups
    expects."""
    return [list(range(start, stop)) for start, stop in group_slices(parts).values()]


def get_funnybirds(
    train: bool,
    batch_size: int,
    img_size: int = 256,
    resampling: bool = True,
    pkls_dir: Optional[str] = None,
    funnybirds_root: Optional[str] = None,
    n_classes: int = 50,
    **kwargs,
):
    """Mirrors src/datasets/cub200.get_cub200's signature and return shape:
    (dataloader, model_kwargs, attr_groups)."""
    if pkls_dir is None:
        raise ValueError("get_funnybirds requires pkls_dir (set via config data.pkls_dir)")
    if funnybirds_root is None:
        raise ValueError(
            "get_funnybirds requires funnybirds_root (set via config "
            "data.funnybirds_root) -- needed to read the official parts.json "
            "for the concept-group structure"
        )

    if train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
        ])

    dataset = FunnyBirdsMCBM(train=train, pkls_dir=pkls_dir, funnybirds_root=funnybirds_root,
                             n_classes=n_classes, transform=transform)

    if train and resampling:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model_kwargs = {
        "n_concepts": dataset.n_concepts,
        "dim_y": n_classes,
        "dim_c": 1,
        "continuous_y": False,
        "continuous_c": False,
        "ch_in": 3,
        "image_size": img_size,
        "imbalance_ratio": find_class_imbalance(dataset.data, multiple_attr=True),
    }
    return dataloader, model_kwargs, dataset.concepts_groups
