"""FunnyBirds loader for the OFFICIAL minimal_cbm trainer.

minimal_cbm's src/datasets/get_loader only knows CUB200/CIFAR10/disentanglement/
CelebA/Spirals. FunnyBirds is not registered. Rather than edit the submodule
(it must stay a verbatim, citable copy), this module provides a loader with the
SAME return contract as src.datasets.cub200.get_cub200, and train/run_mcbm.py
monkeypatches src.datasets.get_loader to dispatch dataset=="FUNNYBIRDS" here.

Return contract (matches get_cub200):
    dataloader, model_kwargs, attr_groups
where each batch is (image, task, concepts) or, with return_nuisances=True,
(image, task, concepts, nuisances_task, nuisances_nontask). FunnyBirds has no
nuisance attributes, so both nuisance tensors are empty (the trainer's nuisance
leakage loop then no-ops; we measure leakage ourselves downstream).

Input: the CBM pickled-lists produced by
data/funnybirds/build_funnybirds_cbm_data.py, i.e.
    $CURATED_DATA/funnybirds_processed/{train,test}.pkl
each a list[dict] with keys img_path, image, class_label, attribute_label,
attribute_certainty (attribute_label is the 26-d one-hot concept vector).
"""
from __future__ import annotations
import os
import pickle
from typing import Optional, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# reuse the official imbalance helper so pos_weight matches the CUB path exactly
from src.datasets.cub200 import find_class_imbalance

N_CONCEPTS = 26
N_CLASSES = 50
# ImageNet stats: torchvision pretrained resnet/inception expect these. (The
# minimal_cbm CUB config uses mean .5/std 2, an unusual choice tied to its own
# recipe; for a pretrained resnet18 the ImageNet stats are the correct default.)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class FunnyBirds(Dataset):
    """Lazy-loading FunnyBirds dataset in the CBM pickled-list schema.

    Unlike minimal_cbm's CUB200 (which reads every image into RAM up front),
    this opens images on access -- FunnyBirds train is 50k images and eager
    loading would blow up host memory on a cluster node.
    """

    def __init__(
        self,
        train: bool,
        pkls_dir: str,
        transform: Optional[Callable] = None,
        return_nuisances: bool = False,
        **kwargs,
    ) -> None:
        self.train = train
        self.return_nuisances = return_nuisances
        self.transform = transform
        pkl = os.path.join(pkls_dir, "train.pkl" if train else "test.pkl")
        with open(pkl, "rb") as f:
            self.data = pickle.load(f)
        assert self.data, f"empty pickle: {pkl}"
        n = len(self.data[0]["attribute_label"])
        assert n == N_CONCEPTS, f"expected {N_CONCEPTS} concepts, pickle has {n}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        rec = self.data[index]
        img = rec.get("img")
        if img is None:
            img = Image.open(rec["img_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        task = rec["class_label"]
        concepts = torch.tensor(rec["attribute_label"]).float()
        if self.return_nuisances:
            empty = torch.tensor([]).float()
            return img, task, concepts, empty, empty
        return img, task, concepts


def get_funnybirds(
    train: bool,
    batch_size: int,
    pkls_dir: str,
    img_size: int = 224,
    resampling: bool = True,
    return_nuisances: bool = False,
    seed: int = 42,
    **kwargs,
):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    dataset = FunnyBirds(
        train=train, pkls_dir=pkls_dir, transform=transform,
        return_nuisances=return_nuisances,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=train, drop_last=train, num_workers=4, pin_memory=True,
    )
    model_kwargs = {
        "n_concepts": N_CONCEPTS,
        "dim_y": N_CLASSES,
        "dim_c": 1,
        "continuous_y": False,
        "continuous_c": False,
        "ch_in": 3,
        "image_size": img_size,
        "imbalance_ratio": find_class_imbalance(dataset.data, True),
    }
    attr_groups = None  # ignored by TrainExperiment; groups live in concepts.json
    return dataloader, model_kwargs, attr_groups
