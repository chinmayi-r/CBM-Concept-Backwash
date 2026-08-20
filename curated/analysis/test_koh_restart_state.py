#!/usr/bin/env python3
"""Prove that Koh's atomic state reproduces uninterrupted deterministic steps."""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build():
    model = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Dropout(0.25), nn.Linear(7, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    return model, optimizer, scheduler


def step(model, optimizer, scheduler) -> None:
    # Exercise all three RNG families captured by the restart patch.
    scale = 0.5 + random.random() + float(np.random.random())
    x = torch.randn(4, 5) * scale
    target = torch.tensor([0, 1, 0, 1])
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), target)
    loss.backward()
    optimizer.step()
    scheduler.step()


def assert_nested_equal(left, right, path="state") -> None:
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}")
        for key in left:
            assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested_equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koh-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.koh_root))
    from CUB.train import _save_restart_state

    seed_all(91)
    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler = build()
    initial_model = {key: value.clone() for key, value in uninterrupted_model.state_dict().items()}
    for _ in range(4):
        step(uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler)

    seed_all(91)
    interrupted_model, interrupted_optimizer, interrupted_scheduler = build()
    interrupted_model.load_state_dict(initial_model)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "restart_state.pth"
        for _ in range(2):
            step(interrupted_model, interrupted_optimizer, interrupted_scheduler)
        _save_restart_state(
            str(path), interrupted_model, interrupted_optimizer, interrupted_scheduler,
            epoch=1, best_val_epoch=1, best_val_acc=42.0, complete=False,
        )
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")

        resumed_model, resumed_optimizer, resumed_scheduler = build()
        resumed_model.load_state_dict(state["model_state_dict"])
        resumed_optimizer.load_state_dict(state["optimizer_state_dict"])
        resumed_scheduler.load_state_dict(state["scheduler_state_dict"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        for _ in range(2):
            step(resumed_model, resumed_optimizer, resumed_scheduler)

    assert_nested_equal(uninterrupted_model.state_dict(), resumed_model.state_dict(), "model")
    assert_nested_equal(
        uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict(), "optimizer"
    )
    assert_nested_equal(
        uninterrupted_scheduler.state_dict(), resumed_scheduler.state_dict(), "scheduler"
    )
    print("[KOH RESTART EQUIVALENCE PASS] uninterrupted == interrupted+resumed")


if __name__ == "__main__":
    main()
