#!/usr/bin/env python3
"""GPU lifecycle test for the production accelerated Koh trainer.

The test enters ``_accelerated_train`` itself. It proves that a fresh run keeps
pre-training manifests, an epoch-boundary interruption resumes exactly, and the
resumed final model equals an uninterrupted final model.
"""
from __future__ import annotations

import os
import random
import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


CURATED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURATED / "compat"))
import koh_accelerated_training as accelerated


class Meter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, count: int) -> None:
        number = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        self.sum += number * count
        self.count += count
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, path: str) -> None:
        self.stream = open(path, "w", encoding="utf-8")

    def write(self, value: str) -> None:
        self.stream.write(value)

    def flush(self) -> None:
        self.stream.flush()


class TinyJoint(torch.nn.Module):
    """Koh-shaped class-plus-26-concept outputs with stochastic dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 8 * 8, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
        )
        self.main_concepts = torch.nn.ModuleList(
            [torch.nn.Linear(32, 1) for _ in range(26)]
        )
        self.aux_concepts = torch.nn.ModuleList(
            [torch.nn.Linear(32, 1) for _ in range(26)]
        )
        self.class_head = torch.nn.Linear(26, 50)

    def _outputs(self, features, heads):
        concepts = [head(features) for head in heads]
        return [self.class_head(torch.cat(concepts, dim=1)), *concepts]

    def forward(self, inputs):
        features = self.features(inputs)
        return (
            self._outputs(features, self.main_concepts),
            self._outputs(features, self.aux_concepts),
        )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(outputs, labels, topk=(1,)):
    del topk
    predicted = outputs.argmax(dim=1).detach().cpu()
    return (100.0 * (predicted == labels).float().mean(),)


def make_args(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        exp="Joint", epochs=accelerated.EPOCHS,
        batch_size=accelerated.BATCH_SIZE, optimizer="sgd",
        lr=accelerated.MAX_LR, attr_loss_weight=0.01,
        weight_decay=0.0004, n_attributes=26, n_class_attr=2,
        ckpt="1", use_aux=True, use_attr=True, end2end=True,
        normalize_loss=True, weighted_loss="multiple",
        use_sigmoid=False, use_relu=False, bottleneck=False, no_img=False,
        uncertain_labels=False, resampling=False, data_dir="data",
        image_dir="images", log_dir=str(output), seed=1,
    )


def make_train_module(root: Path, batch):
    return SimpleNamespace(
        BASE_DIR=str(root), AverageMeter=Meter, Logger=Logger, accuracy=accuracy,
        find_class_imbalance=lambda path, multiple: [1.0] * 26,
        load_data=lambda *positional, **keywords: [batch],
    )


def assert_loss_equivalence(koh_train, batch) -> None:
    """Compare accelerated AMP loss with Koh's official Joint loss on one batch."""
    seed_all(2026)
    reference_model = TinyJoint().cuda()
    accelerated_model = copy.deepcopy(reference_model)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.0)
    accelerated_optimizer = torch.optim.SGD(accelerated_model.parameters(), lr=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    attr_criterion = [
        torch.nn.BCEWithLogitsLoss(weight=torch.tensor([1.0], device="cuda"))
        for _ in range(26)
    ]
    run_args = make_args(Path("unused"))
    reference_loss = koh_train.AverageMeter()
    reference_accuracy = koh_train.AverageMeter()
    seed_all(3030)
    koh_train.run_epoch(
        reference_model, reference_optimizer, [batch], reference_loss,
        reference_accuracy, criterion, attr_criterion, run_args,
        is_training=True,
    )
    seed_all(3030)
    accelerated_loss, _ = accelerated._joint_epoch(
        koh_train, accelerated_model, accelerated_optimizer, [batch], criterion,
        attr_criterion, run_args, torch.cuda.amp.GradScaler(enabled=True),
    )
    difference = abs(float(reference_loss.avg) - float(accelerated_loss.avg))
    if difference > 0.01:
        raise SystemExit(
            "ERROR: accelerated Joint loss diverges from Koh on the same batch: "
            f"koh={reference_loss.avg} accelerated={accelerated_loss.avg} "
            f"difference={difference}"
        )


def prepare_output(path: Path) -> dict[str, bytes]:
    path.mkdir(parents=True)
    payloads = {
        "TRAINING_PROTOCOL.json": b'{"protocol":"accelerated_v1"}\n',
        "MODEL_PREFLIGHT.json": b'{"framework":"koh_joint"}\n',
        "INPUT_INTEGRITY.json": b'{"inputs":"unchanged"}\n',
    }
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    return payloads


def assert_manifests(path: Path, payloads: dict[str, bytes]) -> None:
    changed = [name for name, payload in payloads.items()
               if not (path / name).is_file() or (path / name).read_bytes() != payload]
    if changed:
        raise SystemExit(f"ERROR: trainer changed staging manifests: {changed}")


def model_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    return {name: value.detach().cpu().clone()
            for name, value in saved.state_dict().items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: accelerated lifecycle test requires an allocated GPU")

    original_epochs = accelerated.EPOCHS
    original_milestones = accelerated.MILESTONES
    original_save_restart = accelerated._save_restart
    old_backup = os.environ.pop("KOH_RESTART_BACKUP_DIR", None)
    accelerated.EPOCHS = 2
    accelerated.MILESTONES = (1, 2)
    koh_root = CURATED / "external" / "ConceptBottleneck"
    sys.path.insert(0, str(koh_root))
    import CUB.train as koh_train

    generator = torch.Generator().manual_seed(909)
    inputs = torch.randn(8, 3, 8, 8, generator=generator)
    labels = torch.randint(0, 50, (8,), generator=generator)
    attributes = torch.randint(0, 2, (8, 26), generator=generator)
    batch = (inputs, labels, [attributes[:, index] for index in range(26)])
    assert_loss_equivalence(koh_train, batch)

    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_output, resumed_output = root / "full", root / "resumed"
            full_manifests = prepare_output(full_output)
            resumed_manifests = prepare_output(resumed_output)
            module = make_train_module(root, batch)

            seed_all(1729)
            accelerated._accelerated_train(
                module, TinyJoint(), make_args(full_output)
            )
            assert_manifests(full_output, full_manifests)

            seed_all(1729)
            interrupted_model = TinyJoint()

            def save_then_interrupt(*positional, **keywords):
                original_save_restart(*positional, **keywords)
                epoch = positional[5] if len(positional) > 5 else keywords["epoch"]
                if epoch == 0:
                    raise RuntimeError("SIMULATED_EPOCH_BOUNDARY_INTERRUPT")

            accelerated._save_restart = save_then_interrupt
            try:
                accelerated._accelerated_train(
                    module, interrupted_model, make_args(resumed_output)
                )
            except RuntimeError as error:
                if str(error) != "SIMULATED_EPOCH_BOUNDARY_INTERRUPT":
                    raise
            else:
                raise SystemExit("ERROR: simulated interruption did not occur")
            assert_manifests(resumed_output, resumed_manifests)
            if not (resumed_output / "restart_state.pth").is_file():
                raise SystemExit("ERROR: interrupted run wrote no restart state")

            accelerated._save_restart = original_save_restart
            accelerated._accelerated_train(
                module, TinyJoint(), make_args(resumed_output)
            )
            assert_manifests(resumed_output, resumed_manifests)

            required = (
                "milestone_epoch_001.pth", "milestone_epoch_002.pth",
                "final_model_1.pth", "restart_state.pth",
            )
            missing = [name for name in required
                       if not (resumed_output / name).is_file()]
            if missing:
                raise SystemExit(f"ERROR: resumed outputs missing: {missing}")

            full_state = model_state(full_output / "final_model_1.pth")
            resumed_state = model_state(resumed_output / "final_model_1.pth")
            mismatches = [name for name in full_state
                          if not torch.equal(full_state[name], resumed_state[name])]
            if mismatches:
                raise SystemExit(
                    f"ERROR: uninterrupted/resumed model mismatch: {mismatches}"
                )

            restart = accelerated._load_restart(resumed_output / "restart_state.pth")
            if restart is None or not restart["training_complete"] or restart["next_epoch"] != 2:
                raise SystemExit("ERROR: final restart state is not complete at epoch 2")
    finally:
        accelerated.EPOCHS = original_epochs
        accelerated.MILESTONES = original_milestones
        accelerated._save_restart = original_save_restart
        if old_backup is not None:
            os.environ["KOH_RESTART_BACKUP_DIR"] = old_backup

    print(
        "[KOH ACCELERATED LIFECYCLE PASS] manifests preserved; "
        "interrupted+resumed == uninterrupted"
    )


if __name__ == "__main__":
    main()
