# models/cbm_cub.py

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class CBMConfig:
    num_concepts: int
    num_classes: int = 200
    backbone_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False


class SimpleCBM(nn.Module):
    """
    Simple sequential CBM:
        image -> backbone -> feature vector
               -> concept logits (K)
               -> species logits (C) from concept logits

    - backbone: ResNet-18 or ResNet-50
    - concept_head: linear layer (features -> K concepts)
    - label_head: linear layer (K concepts -> 200 species)

    This mirrors the conceptual pipeline you described:
        parts -> species
    """

    def __init__(self, cfg: CBMConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Backbone
        if cfg.backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
            )
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif cfg.backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if cfg.pretrained else None
            )
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone_name}")

        self.backbone = backbone

        # Optionally freeze backbone
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. Concept head: feature -> K concepts
        self.concept_head = nn.Linear(feat_dim, cfg.num_concepts)

        # 3. Label head: concepts -> 200 species
        self.label_head = nn.Linear(cfg.num_concepts, cfg.num_classes)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)                 # [B, feat_dim]
        concept_logits = self.concept_head(feats)  # [B, K]
        label_logits = self.label_head(concept_logits)  # [B, C]

        if return_features:
            return label_logits, concept_logits, feats
        return label_logits, concept_logits
