#!/usr/bin/env python3
"""Opt-in accelerated trainer for the Koh-style FunnyBird Joint CBM.

This module changes training mechanics, not model construction or the Joint
loss.  It is installed only when ``KOH_TRAINING_PROTOCOL=accelerated_v1``.
The original Koh path remains the default.
"""
from __future__ import annotations

import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL = "accelerated_v1"
BASE_EPOCHS = 100
ALLOWED_TARGET_EPOCHS = (100, 125, 150, 175, 200)
try:
    EPOCHS = int(os.environ.get("KOH_ACCELERATED_TARGET_EPOCHS", BASE_EPOCHS))
except ValueError as error:
    raise RuntimeError("KOH_ACCELERATED_TARGET_EPOCHS must be an integer") from error
if EPOCHS not in ALLOWED_TARGET_EPOCHS:
    raise RuntimeError(
        "KOH_ACCELERATED_TARGET_EPOCHS must be one of "
        f"{ALLOWED_TARGET_EPOCHS}, got {EPOCHS}"
    )
BATCH_SIZE = 128
START_LR = 0.001
MAX_LR = 0.02
MIN_LR = 0.00002
WARMUP_EPOCHS = 5
NUM_WORKERS = 8
MILESTONES = tuple(range(25, EPOCHS + 1, 25))
RESTART_FORMAT = "koh_accelerated_epoch_boundary_v1"


def lr_multiplier(epoch: int) -> float:
    """Warm from START_LR to MAX_LR, then cosine-decay to MIN_LR."""
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    start = START_LR / MAX_LR
    floor = MIN_LR / MAX_LR
    if epoch < WARMUP_EPOCHS:
        if WARMUP_EPOCHS == 1:
            return 1.0
        fraction = epoch / (WARMUP_EPOCHS - 1)
        return start + fraction * (1.0 - start)
    # The accepted 100-epoch schedule is immutable. A declared convergence
    # extension resumes at its terminal learning rate instead of stretching
    # or replaying the original cosine schedule.
    if epoch >= BASE_EPOCHS:
        return floor
    progress = (epoch - WARMUP_EPOCHS) / (
        BASE_EPOCHS - 1 - WARMUP_EPOCHS
    )
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))


def protocol_manifest() -> dict[str, Any]:
    return {
        "status": "PASS",
        "training_protocol": PROTOCOL,
        "epochs": EPOCHS,
        "base_schedule_epochs": BASE_EPOCHS,
        "target_epochs": EPOCHS,
        "continuation": EPOCHS > BASE_EPOCHS,
        "continuation_lr": MIN_LR if EPOCHS > BASE_EPOCHS else None,
        "batch_size": BATCH_SIZE,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0004,
        "start_lr": START_LR,
        "max_lr": MAX_LR,
        "min_lr": MIN_LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "scheduler": "linear_warmup_then_cosine",
        "amp": True,
        "num_workers": NUM_WORKERS,
        "milestone_epochs": list(MILESTONES),
        "restart_format": RESTART_FORMAT,
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _optimizer_to_cuda(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cuda()


def _save_restart(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_accuracy: float,
    complete: bool,
) -> None:
    state = {
        "format": RESTART_FORMAT,
        "training_protocol": PROTOCOL,
        "next_epoch": epoch + 1,
        "best_val_epoch": best_epoch,
        "best_val_acc": float(best_accuracy),
        "training_complete": bool(complete),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    _atomic_torch_save(state, path)
    backup_dir = os.environ.get("KOH_RESTART_BACKUP_DIR", "")
    if backup_dir:
        backup = Path(backup_dir) / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        temporary = backup.with_name(backup.name + ".tmp")
        shutil.copy2(path, temporary)
        os.replace(temporary, backup)


def _load_restart(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state.get("format") != RESTART_FORMAT:
        raise RuntimeError(
            f"restart format {state.get('format')!r} is not {RESTART_FORMAT!r}"
        )
    if state.get("training_protocol") != PROTOCOL:
        raise RuntimeError("restart training protocol mismatch")
    return state


def _install_loader_workers(train_module: Any) -> None:
    dataset_module = __import__("CUB.dataset", fromlist=["DataLoader"])
    original = dataset_module.DataLoader
    if getattr(original, "_koh_accelerated_v1", False):
        return

    def accelerated_loader(*args: Any, **kwargs: Any):
        kwargs.setdefault("num_workers", NUM_WORKERS)
        kwargs.setdefault("pin_memory", True)
        kwargs.setdefault("persistent_workers", False)
        return original(*args, **kwargs)

    accelerated_loader._koh_accelerated_v1 = True
    dataset_module.DataLoader = accelerated_loader


def _joint_epoch(
    train_module: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Any,
    criterion: torch.nn.Module,
    attr_criterion: list[torch.nn.Module],
    args: Any,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[Any, Any]:
    loss_meter = train_module.AverageMeter()
    accuracy_meter = train_module.AverageMeter()
    model.train()

    for inputs, labels, attr_labels in loader:
        attr_labels = torch.stack([value.long() for value in attr_labels]).t()
        inputs_cuda = inputs.cuda(non_blocking=True)
        labels_cuda = labels.cuda(non_blocking=True)
        attrs_cuda = attr_labels.float().cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs, aux_outputs = model(inputs_cuda)
            losses = [
                criterion(outputs[0], labels_cuda)
                + 0.4 * criterion(aux_outputs[0], labels_cuda)
            ]
            for index, attr_loss in enumerate(attr_criterion):
                losses.append(
                    args.attr_loss_weight
                    * (
                        attr_loss(outputs[index + 1].squeeze(), attrs_cuda[:, index])
                        + 0.4
                        * attr_loss(
                            aux_outputs[index + 1].squeeze(), attrs_cuda[:, index]
                        )
                    )
                )
            total_loss = (losses[0] + sum(losses[1:])) / (
                1 + args.attr_loss_weight * args.n_attributes
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"non-finite accelerated Joint loss: {total_loss.detach().cpu().item()}"
            )
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accuracy = train_module.accuracy(outputs[0], labels, topk=(1,))[0]
        loss_meter.update(total_loss.detach().float().item(), inputs.size(0))
        accuracy_meter.update(accuracy, inputs.size(0))

    return loss_meter, accuracy_meter


def _accelerated_train(train_module: Any, model: torch.nn.Module, args: Any) -> None:
    expected = {
        "exp": "Joint",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "sgd",
        "lr": MAX_LR,
        "attr_loss_weight": 0.01,
        "weight_decay": 0.0004,
        "n_attributes": 26,
        "n_class_attr": 2,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"accelerated protocol argument mismatch expected={expected} observed={observed}"
        )
    if not (
        args.ckpt
        and args.use_aux
        and args.use_attr
        and args.end2end
        and args.normalize_loss
        and args.weighted_loss == "multiple"
    ):
        raise RuntimeError(
            "accelerated protocol requires Joint -ckpt/use_aux/use_attr/end2end/"
            "normalize_loss/weighted_loss=multiple"
        )
    if args.use_sigmoid or args.use_relu or args.bottleneck or args.no_img:
        raise RuntimeError("accelerated protocol rejected a non-raw-logit Joint variant")
    if args.uncertain_labels or args.resampling:
        raise RuntimeError(
            "accelerated protocol rejected uncertain labels or resampling"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("accelerated protocol requires CUDA")

    _install_loader_workers(train_module)
    train_path = os.path.join(train_module.BASE_DIR, args.data_dir, "train.pkl")
    val_path = train_path.replace("train.pkl", "val.pkl")
    imbalance = train_module.find_class_imbalance(train_path, True)
    output = Path(args.log_dir)
    restart_path = output / "restart_state.pth"
    restart = _load_restart(restart_path)

    # The staging layer owns this directory and writes protocol, model, and
    # input-integrity manifests before training. Never reproduce Koh's broad
    # log-directory cleanup here. The stage already refuses an unexplained
    # nonempty output without a restart state.
    output.mkdir(parents=True, exist_ok=True)
    if restart is not None and (output / "log.txt").is_file():
        resumed = restart["next_epoch"]
        count = os.environ.get("SLURM_RESTART_COUNT", "unknown")
        job = os.environ.get("SLURM_JOB_ID", "manual")
        shutil.move(
            output / "log.txt",
            output / f"log.before_resume_job_{job}_{count}_epoch_{resumed}.txt",
        )

    logger = train_module.Logger(str(output / "log.txt"))
    logger.write(str(args) + "\n")
    logger.write(str(imbalance) + "\n")
    logger.write(str(protocol_manifest()) + "\n")
    logger.flush()

    if restart is not None:
        model.load_state_dict(restart["model_state_dict"])
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    attr_criterion = [
        torch.nn.BCEWithLogitsLoss(weight=torch.tensor([ratio], device="cuda"))
        for ratio in imbalance
    ]
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=MAX_LR,
        momentum=0.9,
        weight_decay=0.0004,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loader = train_module.load_data(
        [train_path, val_path],
        args.use_attr,
        args.no_img,
        args.batch_size,
        args.uncertain_labels,
        image_dir=args.image_dir,
        n_class_attr=args.n_class_attr,
        resampling=args.resampling,
    )

    start_epoch = 0
    best_epoch = -1
    best_accuracy = 0.0
    if restart is not None:
        optimizer.load_state_dict(restart["optimizer_state_dict"])
        _optimizer_to_cuda(optimizer)
        scheduler.load_state_dict(restart["scheduler_state_dict"])
        scaler.load_state_dict(restart["scaler_state_dict"])
        start_epoch = restart["next_epoch"]
        best_epoch = restart["best_val_epoch"]
        best_accuracy = restart["best_val_acc"]
        random.setstate(restart["python_rng_state"])
        np.random.set_state(restart["numpy_rng_state"])
        torch.set_rng_state(restart["torch_rng_state"])
        torch.cuda.set_rng_state_all(restart["cuda_rng_state_all"])
        logger.write(f"Resuming complete epoch {start_epoch - 1} at epoch {start_epoch}\n")
        logger.flush()
        if restart.get("training_complete", False) and start_epoch >= EPOCHS:
            logger.write("Training was already complete; skipping epoch loop\n")
            logger.flush()
            return
        if restart.get("training_complete", False):
            if start_epoch < BASE_EPOCHS:
                raise RuntimeError(
                    "restart is marked complete before the immutable base schedule"
                )
            logger.write(
                f"Declared convergence continuation: epoch {start_epoch} "
                f"to {EPOCHS} at LR {MIN_LR}\n"
            )
            logger.flush()

    for epoch in range(start_epoch, EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        loss_meter, accuracy_meter = _joint_epoch(
            train_module,
            model,
            optimizer,
            loader,
            criterion,
            attr_criterion,
            args,
            scaler,
        )
        accuracy_value = float(accuracy_meter.avg)
        if accuracy_value > best_accuracy:
            best_epoch = epoch
            best_accuracy = accuracy_value
            _atomic_torch_save(model, output / f"best_model_{args.seed}.pth")

        completed_epoch = epoch + 1
        logger.write(
            "Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t"
            "Combined train+val loss: %.4f\tCombined train+val acc: %.4f\t"
            "Best training epoch: %d\tLR: %.8f\n"
            % (
                epoch,
                loss_meter.avg,
                accuracy_value,
                loss_meter.avg,
                accuracy_value,
                best_epoch,
                current_lr,
            )
        )
        logger.flush()

        if completed_epoch in MILESTONES:
            _atomic_torch_save(
                model, output / f"milestone_epoch_{completed_epoch:03d}.pth"
            )
        if completed_epoch == EPOCHS:
            _atomic_torch_save(model, output / f"final_model_{args.seed}.pth")

        scheduler.step()
        _save_restart(
            restart_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_epoch,
            best_accuracy,
            completed_epoch == EPOCHS,
        )

    print(
        f"[KOH ACCELERATED TRAINING COMPLETE] epochs={EPOCHS} "
        f"best_training_epoch={best_epoch} best_training_accuracy={best_accuracy:.4f}"
    )


def install(train_module: Any) -> None:
    if os.environ.get("KOH_TRAINING_PROTOCOL") != PROTOCOL:
        raise RuntimeError("accelerated trainer installed without accelerated_v1 opt-in")
    if getattr(train_module, "_koh_accelerated_protocol", None):
        raise RuntimeError("Koh training module was already replaced")

    original_train = train_module.train

    def train(model: torch.nn.Module, args: Any) -> None:
        return _accelerated_train(train_module, model, args)

    train._koh_original_train = original_train
    train_module.train = train
    train_module._koh_accelerated_protocol = PROTOCOL
    print(
        "[KOH ACCELERATED INSTALL PASS] "
        f"protocol={PROTOCOL} epochs={EPOCHS} batch={BATCH_SIZE} "
        f"lr={START_LR}->{MAX_LR}->{MIN_LR} amp=1 workers={NUM_WORKERS}"
    )
