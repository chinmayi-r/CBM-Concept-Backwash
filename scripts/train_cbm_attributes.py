# scripts/train_probes_attributes.py

#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch

# cub_metadata lives in top-level "datasets"
from datasets.cub_metadata import load_cub_metadata
# probe_utils is under src/probes
from src.probes.probe_utils import train_linear_probe, ProbeMetrics


def choose_layers() -> List[str]:
    # Same set as your feature extractor
    return ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        type=str,
        default="features/resnet50_cub",
        help="Directory with *_train.pt, *_test.pt, and labels_{train,test}.pt",
    )
    parser.add_argument(
        "--cub_root",
        type=str,
        required=True,
        help="Path to CUB_200_2011 root (the one with attributes/, images/, metadata/)",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="results/probes/resnet50_cub_probes_attributes.json",
        help="Where to write probe results as JSON",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load labels & image ids (species)
    # -------------------------------------------------------------------------
    labels_train = torch.load(feat_dir / "labels_train.pt", weights_only=True)
    labels_test = torch.load(feat_dir / "labels_test.pt", weights_only=True)

    y_train_species = labels_train["labels"]          # shape [N_train]
    y_test_species = labels_test["labels"]            # shape [N_test]
    img_ids_train = labels_train["image_ids"].numpy() # shape [N_train]
    img_ids_test = labels_test["image_ids"].numpy()   # shape [N_test]

    num_species = int(y_train_species.max().item() + 1)

    # -------------------------------------------------------------------------
    # 2. Load CUB metadata, including attribute tables
    # -------------------------------------------------------------------------
    cub_root = Path(args.cub_root)
    meta = load_cub_metadata(cub_root)

    attrs_df = meta.attributes
    img_attr_binary = None
    if meta.image_attributes_binary is not None:
        img_attr_binary = meta.image_attributes_binary.set_index("image_id")

    if attrs_df is None or img_attr_binary is None:
        raise RuntimeError(
            "Attribute metadata not available. "
            "Expected attributes/attributes.txt and attributes/image_attribute_labels.txt "
            "to be present under cub_root."
        )

    # -------------------------------------------------------------------------
    # 3. Select usable attributes (neither always on nor always off)
    # -------------------------------------------------------------------------
    attr_cols = [c for c in img_attr_binary.columns if c.startswith("attr_")]
    freqs = img_attr_binary[attr_cols].mean(axis=0)

    # Keep attributes with presence between 10% and 90%
    valid_mask = (freqs > 0.10) & (freqs < 0.90)
    chosen_attr_cols = list(freqs[valid_mask].index)

    print(f"[train_probes_attributes] Total attributes: {len(attr_cols)}")
    print(f"[train_probes_attributes] Usable attributes (10%–90% freq): {len(chosen_attr_cols)}")
    print("[train_probes_attributes] Example chosen attributes:", chosen_attr_cols[:10])

    # Map attribute_id -> (name, group)
    attrs_df = attrs_df.set_index("attribute_id")
    attr_name_from_col: Dict[str, str] = {}
    attr_group_from_col: Dict[str, str] = {}

    for col in chosen_attr_cols:
        # attr_10_present -> 10
        attr_id = int(col.split("_")[1])
        row = attrs_df.loc[attr_id]
        attr_name_from_col[col] = row["attribute_name"]  # e.g., has_bill_color::black
        attr_group_from_col[col] = row["group"]         # e.g., has_bill_color

    # Build per-attribute labels aligned with img_ids_train / img_ids_test
    y_train_attrs: Dict[str, torch.Tensor] = {}
    y_test_attrs: Dict[str, torch.Tensor] = {}

    for col in chosen_attr_cols:
        y_train_attrs[col] = torch.tensor(
            [int(img_attr_binary.loc[i, col]) if i in img_attr_binary.index else 0
             for i in img_ids_train],
            dtype=torch.long,
        )
        y_test_attrs[col] = torch.tensor(
            [int(img_attr_binary.loc[i, col]) if i in img_attr_binary.index else 0
             for i in img_ids_test],
            dtype=torch.long,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_probes_attributes] Using device:", device)

    results = []  # list of dicts to be dumped as JSON

    # -------------------------------------------------------------------------
    # 4. Species probes (subclass)
    # -------------------------------------------------------------------------
    for layer_name in choose_layers():
        print(f"[train_probes_attributes] Species probe on layer {layer_name}")
        x_train = torch.load(feat_dir / f"{layer_name}_train.pt", weights_only=True)
        x_test = torch.load(feat_dir / f"{layer_name}_test.pt", weights_only=True)

        metrics: ProbeMetrics = train_linear_probe(
            x_train=x_train,
            y_train=y_train_species,
            x_val=x_test,
            y_val=y_test_species,
            num_classes=num_species,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

        results.append(
            {
                "layer": layer_name,
                "target_type": "subclass",
                "target_name": "species",
                "group": "species",
                "train_acc": metrics.train_acc,
                "val_acc": metrics.val_acc,
                "best_val_acc": metrics.best_val_acc,
            }
        )

    # -------------------------------------------------------------------------
    # 5. Attribute probes (replacing part-concept probes)
    # -------------------------------------------------------------------------
    for layer_name in choose_layers():
        print(f"[train_probes_attributes] Attribute probes on layer {layer_name}")
        x_train = torch.load(feat_dir / f"{layer_name}_train.pt", weights_only=True)
        x_test = torch.load(feat_dir / f"{layer_name}_test.pt", weights_only=True)

        for col in chosen_attr_cols:
            attr_name = attr_name_from_col[col]
            group = attr_group_from_col[col]

            y_train_attr = y_train_attrs[col]
            y_test_attr = y_test_attrs[col]

            metrics: ProbeMetrics = train_linear_probe(
                x_train=x_train,
                y_train=y_train_attr,
                x_val=x_test,
                y_val=y_test_attr,
                num_classes=2,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
            )

            results.append(
                {
                    "layer": layer_name,
                    "target_type": "attribute",
                    "target_name": attr_name,   # e.g. "has_wing_color::brown"
                    "column": col,              # e.g. "attr_10_present"
                    "group": group,             # e.g. "has_wing_color"
                    "train_acc": metrics.train_acc,
                    "val_acc": metrics.val_acc,
                    "best_val_acc": metrics.best_val_acc,
                }
            )

    # -------------------------------------------------------------------------
    # 6. Save JSON
    # -------------------------------------------------------------------------
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"[train_probes_attributes] Wrote attribute + species probe results to {out_path}")


if __name__ == "__main__":
    main()
