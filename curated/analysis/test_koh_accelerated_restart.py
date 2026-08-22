#!/usr/bin/env python3
"""GPU smoke test that AMP/optimizer/scheduler/RNG restart is exact."""
from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


CURATED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURATED / "compat"))
import koh_accelerated_training as accelerated


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def components():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)
    ).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=accelerated.MAX_LR, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, accelerated.lr_multiplier
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    return model, optimizer, scheduler, scaler


def step(model, optimizer, scheduler, scaler):
    inputs = torch.randn(32, 8, device="cuda")
    labels = torch.randint(0, 3, (32,), device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


def state(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: accelerated restart test requires an allocated GPU")

    seed_all(1729)
    full_model, full_optimizer, full_scheduler, full_scaler = components()
    for _ in range(4):
        step(full_model, full_optimizer, full_scheduler, full_scaler)
    full_state = state(full_model)
    full_lr = full_optimizer.param_groups[0]["lr"]
    full_scaler_state = full_scaler.state_dict()

    seed_all(1729)
    model, optimizer, scheduler, scaler = components()
    for _ in range(2):
        step(model, optimizer, scheduler, scaler)

    old_backup = os.environ.pop("KOH_RESTART_BACKUP_DIR", None)
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restart_state.pth"
            accelerated._save_restart(
                path, model, optimizer, scheduler, scaler,
                epoch=1, best_epoch=1, best_accuracy=0.0, complete=False,
            )
            saved = accelerated._load_restart(path)
            assert saved is not None

            resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler = components()
            resumed_model.load_state_dict(saved["model_state_dict"])
            resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])
            accelerated._optimizer_to_cuda(resumed_optimizer)
            resumed_scheduler.load_state_dict(saved["scheduler_state_dict"])
            resumed_scaler.load_state_dict(saved["scaler_state_dict"])
            random.setstate(saved["python_rng_state"])
            np.random.set_state(saved["numpy_rng_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])

            for _ in range(2):
                step(
                    resumed_model, resumed_optimizer,
                    resumed_scheduler, resumed_scaler,
                )
    finally:
        if old_backup is not None:
            os.environ["KOH_RESTART_BACKUP_DIR"] = old_backup

    resumed_state = state(resumed_model)
    mismatches = [name for name in full_state
                  if not torch.equal(full_state[name], resumed_state[name])]
    if mismatches:
        raise SystemExit(f"ERROR: accelerated restart mismatch: {mismatches}")
    if resumed_optimizer.param_groups[0]["lr"] != full_lr:
        raise SystemExit("ERROR: accelerated restart LR mismatch")
    if resumed_scaler.state_dict() != full_scaler_state:
        raise SystemExit("ERROR: accelerated restart scaler mismatch")
    print("[KOH ACCELERATED RESTART EQUIVALENCE PASS]")


if __name__ == "__main__":
    main()
