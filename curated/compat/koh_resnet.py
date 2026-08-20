"""ResNet-50 image-to-concept encoder for Koh's unchanged Joint CBM wrapper.

This module deliberately implements only Koh's ``model1`` interface: an image
encoder that returns one raw scalar logit per concept and, while training, one
matching auxiliary logit per concept.  Koh's existing ``End2EndModel`` remains
responsible for concatenating those raw logits and applying its linear
concept-to-class head.  Nothing from ``minimal_cbm`` is imported here.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


def _concept_heads(input_dim: int, n_attributes: int, expand_dim: int) -> nn.ModuleList:
    """Build Koh-shaped scalar heads without importing another CBM framework."""
    heads = nn.ModuleList()
    for _ in range(n_attributes):
        if expand_dim:
            head = nn.Sequential(
                nn.Linear(input_dim, expand_dim),
                nn.ReLU(),
                nn.Linear(expand_dim, 1),
            )
        else:
            head = nn.Linear(input_dim, 1)
        for module in head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.1, a=-0.2, b=0.2)
        heads.append(head)
    return heads


class KohResNet50ConceptEncoder(nn.Module):
    """Drop-in replacement for Koh's Inception image-to-concept ``model1``.

    The input transformation is copied from Koh's pretrained Inception path so
    the unchanged Koh data loader can continue emitting its historical
    ``Normalize(mean=.5, std=2)`` tensors.  ResNet accepts the same 299x299
    crops; changing crop size is intentionally outside this adapter's scope.
    """

    curated_framework = "koh_joint"
    curated_backbone = "resnet50"

    def __init__(
        self,
        *,
        pretrained: bool,
        freeze: bool,
        num_classes: int,
        use_aux: bool,
        n_attributes: int,
        expand_dim: int,
        three_class: bool,
    ) -> None:
        super().__init__()
        if n_attributes <= 0:
            raise ValueError("Koh ResNet Joint requires at least one concept")
        if three_class:
            raise ValueError("Koh ResNet adapter currently supports binary concepts only")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        self.transform_input = bool(pretrained)
        self.aux_logits = bool(use_aux)
        self.n_attributes = int(n_attributes)
        self.num_classes = int(num_classes)

        # Preserve torchvision's pretrained backbone exactly.  Only its unused
        # ImageNet classifier is discarded and replaced by Koh-shaped concept
        # heads.  Layer 3 supplies the auxiliary concept logits expected by
        # Koh's unchanged ``-use_aux`` loss path.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.main_heads = _concept_heads(2048, n_attributes, expand_dim)
        self.aux_heads = (
            _concept_heads(1024, n_attributes, expand_dim) if use_aux else None
        )

        if freeze:
            for name, parameter in self.named_parameters():
                if not name.startswith(("main_heads", "aux_heads")):
                    parameter.requires_grad = False

    @staticmethod
    def _koh_to_imagenet_input(x: Tensor) -> Tensor:
        # Exact channel-wise conversion used by Koh's pretrained Inception.
        x0 = x[:, 0:1] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x1 = x[:, 1:2] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x2 = x[:, 2:3] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return torch.cat((x0, x1, x2), dim=1)

    @staticmethod
    def _apply_heads(heads: nn.ModuleList, features: Tensor) -> List[Tensor]:
        return [head(features) for head in heads]

    def forward(
        self, x: Tensor
    ) -> Union[List[Tensor], Tuple[List[Tensor], List[Tensor]]]:
        if self.transform_input:
            x = self._koh_to_imagenet_input(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)

        aux_outputs = None
        if self.training and self.aux_logits:
            if self.aux_heads is None:
                raise RuntimeError("auxiliary logits requested but auxiliary heads are missing")
            aux_features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(
                layer3, (1, 1)
            ), 1)
            aux_outputs = self._apply_heads(self.aux_heads, aux_features)

        x = self.layer4(layer3)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.dropout(x, training=self.training)
        outputs = self._apply_heads(self.main_heads, x)
        if self.training and self.aux_logits:
            return outputs, aux_outputs
        return outputs


def build_koh_resnet50_joint(
    n_class_attr: int,
    pretrained: bool,
    freeze: bool,
    num_classes: int,
    use_aux: bool,
    n_attributes: int,
    expand_dim: int,
    use_relu: bool,
    use_sigmoid: bool,
):
    """Construct Koh Joint with only ``model1`` changed to ResNet-50."""
    if n_class_attr != 2:
        raise ValueError("Koh ResNet Joint requires binary concepts")
    if use_relu or use_sigmoid:
        raise ValueError("Koh ResNet Joint class head must read raw concept logits")

    # Import Koh only here so importing this module cannot accidentally select
    # another CBM implementation.
    from CUB.template_model import End2EndModel, MLP

    model1 = KohResNet50ConceptEncoder(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        use_aux=use_aux,
        n_attributes=n_attributes,
        expand_dim=expand_dim,
        three_class=False,
    )
    model2 = MLP(
        input_dim=n_attributes,
        num_classes=num_classes,
        expand_dim=expand_dim,
    )
    model = End2EndModel(
        model1,
        model2,
        use_relu=False,
        use_sigmoid=False,
        n_class_attr=n_class_attr,
    )
    model.curated_framework = "koh_joint"
    model.curated_backbone = "resnet50"
    return model
