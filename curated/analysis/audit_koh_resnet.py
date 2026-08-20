#!/usr/bin/env python3
"""Fail-closed audits for the seed-1 FunnyBird Koh/ResNet adaptation."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import re
import sys
from pathlib import Path


CURATED = Path(__file__).resolve().parents[1]
COMPAT = CURATED / "compat"
KOH = CURATED / "external" / "ConceptBottleneck"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_audit() -> None:
    from torchvision.models import ResNet50_Weights

    filename = Path(ResNet50_Weights.IMAGENET1K_V1.url).name
    cache = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    path = cache / "hub" / "checkpoints" / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(
            f"ERROR: ResNet-50 weights are not cached: {path}\n"
            "Run: python3 curated/analysis/audit_koh_resnet.py fetch-weights"
        )
    print(json.dumps({
        "status": "PASS",
        "weights": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "torchvision_enum": "ResNet50_Weights.IMAGENET1K_V1",
    }, sort_keys=True))


def fetch_weights() -> None:
    from torchvision.models import ResNet50_Weights

    state = ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
    if not state:
        raise SystemExit("ERROR: torchvision returned an empty ResNet state dict")
    weights_audit()


def _install_paths(koh_root: Path) -> None:
    sys.path.insert(0, str(COMPAT))
    sys.path.insert(0, str(koh_root))


def model_audit(koh_root: Path, output: Path | None) -> None:
    import torch
    from torch import nn

    _install_paths(koh_root)
    from koh_resnet import build_koh_resnet50_joint

    model = build_koh_resnet50_joint(
        n_class_attr=2,
        pretrained=False,
        freeze=False,
        num_classes=50,
        use_aux=True,
        n_attributes=26,
        expand_dim=0,
        use_relu=False,
        use_sigmoid=False,
    )
    if type(model).__name__ != "End2EndModel":
        raise SystemExit(f"ERROR: expected Koh End2EndModel, got {type(model).__name__}")
    if model.curated_framework != "koh_joint" or model.curated_backbone != "resnet50":
        raise SystemExit("ERROR: missing Koh/ResNet identity markers")
    if model.use_relu or model.use_sigmoid:
        raise SystemExit("ERROR: class head does not read raw concept logits")
    if len(model.first_model.main_heads) != 26 or len(model.first_model.aux_heads) != 26:
        raise SystemExit("ERROR: expected 26 main and 26 auxiliary scalar concept heads")
    if not all(isinstance(head, nn.Linear) and head.out_features == 1
               for head in model.first_model.main_heads):
        raise SystemExit("ERROR: main concept heads are not scalar linear heads")
    class_head = model.sec_model.linear
    if not isinstance(class_head, nn.Linear) or (
        class_head.in_features, class_head.out_features
    ) != (26, 50):
        raise SystemExit("ERROR: class head is not the required linear 26->50 map")

    model.eval()
    with torch.inference_mode():
        output_values = model(torch.zeros(2, 3, 64, 64))
    if len(output_values) != 27:
        raise SystemExit(f"ERROR: expected class + 26 concept outputs, got {len(output_values)}")
    if output_values[0].shape != (2, 50):
        raise SystemExit(f"ERROR: class output shape is {tuple(output_values[0].shape)}")
    if any(value.shape != (2, 1) for value in output_values[1:]):
        raise SystemExit("ERROR: concept outputs are not scalar per image")

    modules = sorted({type(module).__module__ for module in model.modules()})
    if any("minimal_cbm" in name or name.startswith("src.models") for name in modules):
        raise SystemExit(f"ERROR: MCBM module found in Koh model: {modules}")

    train_source = inspect.getsource(__import__("CUB.train", fromlist=["run_epoch"]).run_epoch)
    required_fragments = (
        "args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start]",
        "+ 0.4 * attr_criterion[i](aux_outputs[i+out_start]",
        "total_loss = losses[0] + sum(losses[1:])",
        "total_loss / (1 + args.attr_loss_weight * args.n_attributes)",
    )
    missing = [fragment for fragment in required_fragments if fragment not in train_source]
    if missing:
        raise SystemExit(f"ERROR: pinned Koh loss source changed; missing={missing}")

    report = {
        "status": "PASS",
        "framework": "koh_joint",
        "backbone": "resnet50",
        "concept_heads": 26,
        "concept_head_output": 1,
        "class_head": [26, 50],
        "use_relu": False,
        "use_sigmoid": False,
        "auxiliary_concept_heads": 26,
        "minimal_cbm_modules": [],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "koh_root": str(koh_root.resolve()),
        "adapter_sha256": sha256(COMPAT / "koh_resnet.py"),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


def _resolve_image(work_dir: Path, raw: str) -> Path:
    parts = raw.replace("\\", "/").split("/")
    if "CUB_200_2011" not in parts:
        raise ValueError(f"image path lacks CUB_200_2011 marker: {raw}")
    index = parts.index("CUB_200_2011")
    return work_dir.joinpath(*parts[index:])


def data_audit(pkls: list[Path], work_dir: Path, output: Path) -> None:
    entries = []
    image_hashes = hashlib.sha256()
    seen: set[Path] = set()
    for pkl_path in pkls:
        records = pickle.loads(pkl_path.read_bytes())
        for record in records:
            image = _resolve_image(work_dir, str(record["img_path"]))
            if image in seen:
                continue
            seen.add(image)
            if not image.is_file():
                raise SystemExit(f"ERROR: missing image referenced by pickle: {image}")
            relative = image.relative_to(work_dir).as_posix()
            image_hashes.update(relative.encode("utf-8") + b"\0")
            image_hashes.update(bytes.fromhex(sha256(image)))
        entries.append({
            "path": str(pkl_path.resolve()),
            "bytes": pkl_path.stat().st_size,
            "sha256": sha256(pkl_path),
            "rows": len(records),
        })
    report = {
        "status": "PASS",
        "pickles": entries,
        "unique_images": len(seen),
        "ordered_image_content_sha256": image_hashes.hexdigest(),
        "work_dir": str(work_dir.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


def launcher_audit(stage: Path, adapter: Path) -> None:
    stage_text = stage.read_text()
    command_match = re.search(r'CMD=\((.*?)\n\)', stage_text, flags=re.DOTALL)
    if not command_match:
        raise SystemExit("ERROR: cannot locate Koh CMD array")
    command = command_match.group(1)
    required = (
        'CUB Joint --seed "$SEED" -ckpt 1',
        '-e 1000 -optimizer sgd -pretrained -use_aux -use_attr',
        '-weighted_loss multiple -data_dir "$DATA"',
        '-n_attributes "$N_ATTR" -attr_loss_weight 0.01 -normalize_loss -b 64',
        '-weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end',
    )
    missing = [value for value in required if value not in command]
    forbidden = [value for value in ("-use_sigmoid", "-use_relu") if value in command]
    if missing or forbidden:
        raise SystemExit(
            f"ERROR: Koh launcher contract mismatch missing={missing} forbidden={forbidden}"
        )
    adapter_text = adapter.read_text()
    if "koh_models.ModelXtoCtoY = build_koh_resnet50_joint" not in adapter_text:
        raise SystemExit("ERROR: ResNet adapter does not replace exactly Koh ModelXtoCtoY")
    if "ModelXtoC =" in adapter_text or "ModelXtoY =" in adapter_text:
        raise SystemExit("ERROR: ResNet adapter patches an unapproved Koh model constructor")
    print(json.dumps({
        "status": "PASS",
        "experiment": "CUB Joint",
        "epochs_max": 1000,
        "optimizer": "sgd",
        "attr_loss_weight": 0.01,
        "batch_size": 64,
        "use_aux": True,
        "use_sigmoid": False,
        "use_relu": False,
        "patched_constructor": "ModelXtoCtoY",
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("weights")
    sub.add_parser("fetch-weights")
    model = sub.add_parser("model")
    model.add_argument("--koh-root", type=Path, default=KOH)
    model.add_argument("--output", type=Path)
    data = sub.add_parser("data")
    data.add_argument("--pkl", type=Path, action="append", required=True)
    data.add_argument("--work-dir", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    launcher = sub.add_parser("launcher")
    launcher.add_argument("--stage", type=Path,
                          default=CURATED / "train" / "koh_joint_stage.sh")
    launcher.add_argument("--adapter", type=Path,
                          default=CURATED / "compat" / "run_koh.py")
    args = parser.parse_args()
    if args.command == "weights":
        weights_audit()
    elif args.command == "fetch-weights":
        fetch_weights()
    elif args.command == "model":
        model_audit(args.koh_root, args.output)
    elif args.command == "data":
        data_audit(args.pkl, args.work_dir, args.output)
    else:
        launcher_audit(args.stage, args.adapter)


if __name__ == "__main__":
    main()
