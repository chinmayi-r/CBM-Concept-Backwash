import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

def get_resnet50_cub(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    """
    Return a ResNet-50 model adapted for CUB (200 classes).
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)

    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_classes)
    return model


def save_checkpoint(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load weights into a ResNet model, allowing missing keys like fc.weight/fc.bias.

    This is important when we load a CBM backbone checkpoint that was saved
    without a fully-connected classification head.
    """
    state = torch.load(path, map_location=device or "cpu", weights_only=True)

    # allow missing / unexpected keys
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print("[load_checkpoint] Missing keys:", missing)
    if unexpected:
        print("[load_checkpoint] Unexpected keys:", unexpected)

    return model
