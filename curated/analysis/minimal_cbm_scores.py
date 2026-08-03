#!/usr/bin/env python3
"""Recover exact minimal_cbm concept logits from saved latent slots.

Prediction files save encoder output ``z`` and activated concept predictions
``c_preds``, but not ``c_logits``. FunnyBird and CUB CBM configs use a learned
per-concept MLP, so saved ``z`` is not itself a concept logit. This module
applies the saved concept heads without rerunning images.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F


_WEIGHT_RE = re.compile(r"(?:^|\.)mlp_c\.(\d+)\.(\d+)\.weight$")


def _as_2d(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu().float()
    return value.reshape(value.shape[0], -1)


def concept_logits_from_saved_latent(
    latent: torch.Tensor,
    checkpoint_path: str | Path,
    n_concepts: int,
) -> torch.Tensor:
    """Return ``[N, J]`` raw logits from saved latent slots and checkpoint."""
    latent = _as_2d(latent)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)

    layers: dict[int, list[tuple[int, str]]] = {j: [] for j in range(n_concepts)}
    for key in state:
        match = _WEIGHT_RE.search(key)
        if match:
            concept, layer = map(int, match.groups())
            if concept < n_concepts:
                layers[concept].append((layer, key))

    if not any(layers.values()):
        if latent.shape[1] != n_concepts:
            raise RuntimeError(
                "checkpoint has no learned concept-head weights, but saved latent "
                f"width {latent.shape[1]} != n_concepts {n_concepts}"
            )
        return latent
    if not all(layers.values()):
        missing = [j for j, found in layers.items() if not found]
        raise RuntimeError(f"checkpoint is missing learned concept heads: {missing}")

    input_widths = [int(state[min(layers[j])[1]].shape[1]) for j in range(n_concepts)]
    if sum(input_widths) != latent.shape[1]:
        raise RuntimeError(
            "saved latent width does not match concept-head inputs: "
            f"latent={latent.shape[1]}, head_inputs={sum(input_widths)}"
        )

    outputs = []
    offset = 0
    for j, width in enumerate(input_widths):
        value = latent[:, offset:offset + width]
        offset += width
        ordered = sorted(layers[j])
        for position, (_, weight_key) in enumerate(ordered):
            bias_key = weight_key[:-len("weight")] + "bias"
            bias = state.get(bias_key)
            value = F.linear(
                value,
                state[weight_key].detach().cpu().float(),
                None if bias is None else bias.detach().cpu().float(),
            )
            if position < len(ordered) - 1:
                value = F.relu(value)
        if value.shape[1] != 1:
            raise RuntimeError(
                f"concept {j} head produced width {value.shape[1]}; expected 1"
            )
        outputs.append(value[:, 0])
    return torch.stack(outputs, dim=1)


def validate_saved_probabilities(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    tolerance: float = 2e-6,
) -> float:
    """Check that recovered logits reproduce saved concept probabilities."""
    probabilities = _as_2d(probabilities)
    logits = _as_2d(logits)
    if logits.shape != probabilities.shape:
        raise RuntimeError(
            f"logit/probability shape mismatch: {tuple(logits.shape)} vs "
            f"{tuple(probabilities.shape)}"
        )
    error = float((torch.sigmoid(logits) - probabilities).abs().max())
    if error > tolerance:
        raise RuntimeError(
            "recovered logits do not reproduce saved probabilities: "
            f"max error={error:.6g} > {tolerance:.6g}"
        )
    return error
