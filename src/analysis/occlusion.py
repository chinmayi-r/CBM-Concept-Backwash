# src/analysis/occlusion.py

from pathlib import Path
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image

from models.resnet_cub import get_resnet50_cub, load_checkpoint


def make_eval_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def load_image(cub_root: str | Path, rel_path: str) -> Image.Image:
    cub_root = Path(cub_root)
    if rel_path.startswith("images/"):
        rel_path = rel_path.split("images/", 1)[1]
    img_path = cub_root / "images" / rel_path
    return Image.open(img_path).convert("RGB")


def occlude_region(image_tensor: torch.Tensor, box: Tuple[int, int, int, int], mode: str = "mean") -> torch.Tensor:
    """
    image_tensor: (3, H, W)
    box: (x, y, w, h) in pixel coordinates (in the transformed space)
    mode: "mean" or "zero"
    """
    x, y, w, h = box
    img = image_tensor.clone()
    if mode == "zero":
        img[:, y:y+h, x:x+w] = 0.0
    elif mode == "mean":
        patch = img[:, y:y+h, x:x+w]
        if patch.numel() > 0:
            patch_mean = patch.mean(dim=[1, 2], keepdim=True)
            img[:, y:y+h, x:x+w] = patch_mean
    return img


def compute_logits(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    image_tensor: shape (3, H, W)
    Returns logits vector shape (num_classes,).
    """
    model.eval()
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        logits = model(x)
    return logits.squeeze(0).cpu()
