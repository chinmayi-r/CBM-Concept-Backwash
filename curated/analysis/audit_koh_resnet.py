#!/usr/bin/env python3
"""Required behavioral audits for the FunnyBird Koh/ResNet boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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
        raise SystemExit(f"ERROR: missing ResNet-50 ImageNet-V1 weights: {path}")
    observed = sha256(path)
    expected = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
    if observed != expected:
        raise SystemExit(
            f"ERROR: ResNet-50 weight hash {observed} does not match {expected}"
        )
    print(json.dumps({
        "status": "PASS",
        "weights": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": observed,
        "torchvision_enum": "ResNet50_Weights.IMAGENET1K_V1",
    }, sort_keys=True))


def _install_paths(koh_root: Path) -> None:
    sys.path.insert(0, str(COMPAT))
    sys.path.insert(0, str(koh_root))


def model_audit(
    koh_root: Path,
    output: Path | None,
    num_classes: int,
    num_attributes: int,
) -> None:
    import torch
    from torch import nn

    _install_paths(koh_root)
    from koh_resnet import KohResNet50ConceptEncoder, build_koh_resnet50_joint

    model = build_koh_resnet50_joint(
        n_class_attr=2,
        pretrained=False,
        freeze=False,
        num_classes=num_classes,
        use_aux=True,
        n_attributes=num_attributes,
        expand_dim=0,
        use_relu=False,
        use_sigmoid=False,
    )
    concept_encoder = model.first_model
    class_head = model.sec_model.linear
    module_names = [
        f"{type(module).__module__}.{type(module).__name__}"
        for module in model.modules()
    ]
    predicates = {
        "koh_end_to_end": type(model).__name__ == "End2EndModel",
        "resnet_encoder_type": isinstance(
            concept_encoder, KohResNet50ConceptEncoder
        ),
        "framework": getattr(model, "curated_framework", None) == "koh_joint",
        "backbone": getattr(model, "curated_backbone", None) == "resnet50",
        "no_inception_module": not any(
            "inception" in name.lower() for name in module_names
        ),
        "no_minimal_cbm_module": not any(
            "minimal_cbm" in name.lower() or ".mcbm" in name.lower()
            for name in module_names
        ),
        "raw_class_input": not model.use_relu and not model.use_sigmoid,
        "main_scalar_heads": len(concept_encoder.main_heads) == num_attributes and all(
            isinstance(head, nn.Linear) and head.out_features == 1
            for head in concept_encoder.main_heads
        ),
        "aux_scalar_heads": len(concept_encoder.aux_heads) == num_attributes and all(
            isinstance(head, nn.Linear) and head.out_features == 1
            for head in concept_encoder.aux_heads
        ),
        "linear_class_head": isinstance(class_head, nn.Linear)
        and (class_head.in_features, class_head.out_features)
        == (num_attributes, num_classes),
    }
    failed = sorted(name for name, passed in predicates.items() if not passed)
    if failed:
        raise SystemExit(f"ERROR: Koh/ResNet structure predicates failed: {failed}")

    rgb = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1)
    koh_tensor = (rgb - 0.5) / 2.0
    observed = KohResNet50ConceptEncoder._koh_loader_to_resnet_input(koh_tensor)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if not torch.allclose(observed, (rgb - mean) / std, atol=1e-7, rtol=0):
        raise SystemExit("ERROR: Koh-loader to ResNet preprocessing mismatch")

    model.eval()
    with torch.inference_mode():
        values = model(torch.zeros(2, 3, 64, 64))
    if len(values) != num_attributes + 1 or values[0].shape != (2, num_classes) or any(
        value.shape != (2, 1) for value in values[1:]
    ):
        raise SystemExit("ERROR: Koh/ResNet output contract mismatch")

    report = {
        "status": "PASS",
        "framework": "koh_joint",
        "backbone": "resnet50",
        "preprocessing": "invert_koh_mean0.5_std2_then_imagenet1k_v1",
        "concept_heads": num_attributes,
        "concept_head_output": 1,
        "auxiliary_concept_heads": num_attributes,
        "class_head": [num_attributes, num_classes],
        "use_relu": False,
        "use_sigmoid": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "koh_root": str(koh_root.resolve()),
        "adapter_sha256": sha256(COMPAT / "koh_resnet.py"),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(json.dumps(report, sort_keys=True))


def import_boundary_audit(koh_root: Path, num_classes: int) -> None:
    """Prove Koh train copied the patched class count and ResNet constructor."""
    _install_paths(koh_root)
    import CUB.config as config
    import CUB.models as models
    from koh_resnet import build_koh_resnet50_joint

    if "inception_v3" not in models.ModelXtoCtoY.__code__.co_names:
        raise SystemExit("ERROR: unexpected pinned Koh Joint constructor")
    config.N_CLASSES = num_classes
    models.ModelXtoCtoY = build_koh_resnet50_joint
    import CUB.train as train

    minimal_loaded = any(
        name == "minimal_cbm" or name.startswith("minimal_cbm.")
        for name in sys.modules
    )
    predicates = {
        "train_class_count": train.N_CLASSES == num_classes,
        "train_joint_constructor": (
            train.ModelXtoCtoY is build_koh_resnet50_joint
        ),
        "constructor_module": (
            train.ModelXtoCtoY.__module__ == "koh_resnet"
        ),
        "no_minimal_cbm_import": not minimal_loaded,
    }
    failed = sorted(name for name, passed in predicates.items() if not passed)
    if failed:
        raise SystemExit(f"ERROR: Koh import-boundary predicates failed: {failed}")
    print(json.dumps({
        "status": "PASS",
        "framework": "koh_joint",
        "backbone": "resnet50",
        "num_classes": num_classes,
        "train_constructor": train.ModelXtoCtoY.__module__ + "."
        + train.ModelXtoCtoY.__name__,
        "minimal_cbm_imported": minimal_loaded,
    }, sort_keys=True))


def _resolve_image(work_dir: Path, raw: str) -> Path:
    parts = raw.replace("\\", "/").split("/")
    if "CUB_200_2011" not in parts:
        raise ValueError(f"image path lacks CUB_200_2011 marker: {raw}")
    return work_dir.joinpath(*parts[parts.index("CUB_200_2011"):])


def data_audit(pkls: list[Path], work_dir: Path, output: Path) -> None:
    entries = []
    image_digest = hashlib.sha256()
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
            image_digest.update(image.relative_to(work_dir).as_posix().encode() + b"\0")
            image_digest.update(bytes.fromhex(sha256(image)))
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
        "ordered_image_content_sha256": image_digest.hexdigest(),
        "work_dir": str(work_dir.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("weights")
    model = sub.add_parser("model")
    model.add_argument("--koh-root", type=Path, default=KOH)
    model.add_argument("--output", type=Path)
    model.add_argument("--num-classes", type=int, default=50)
    model.add_argument("--num-attributes", type=int, default=26)
    boundary = sub.add_parser("boundary")
    boundary.add_argument("--koh-root", type=Path, default=KOH)
    boundary.add_argument("--num-classes", type=int, required=True)
    data = sub.add_parser("data")
    data.add_argument("--pkl", type=Path, action="append", required=True)
    data.add_argument("--work-dir", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "weights":
        weights_audit()
    elif args.command == "model":
        model_audit(
            args.koh_root, args.output, args.num_classes, args.num_attributes
        )
    elif args.command == "boundary":
        import_boundary_audit(args.koh_root, args.num_classes)
    else:
        data_audit(args.pkl, args.work_dir, args.output)


if __name__ == "__main__":
    main()
