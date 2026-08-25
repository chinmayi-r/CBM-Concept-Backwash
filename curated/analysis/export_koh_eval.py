#!/usr/bin/env python3
"""Export raw concept logits from a pinned Koh checkpoint to Parquet."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_model(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False).to(device).eval()
    except TypeError:
        return torch.load(path, map_location=device).to(device).eval()


def names(args):
    if args.names:
        result = json.loads(Path(args.names).read_text())
    else:
        selected = json.loads(Path(args.selection_indices).read_text())
        lines = [line.split(maxsplit=1)[1] for line in Path(args.attributes).read_text().splitlines() if line.strip()]
        result = [lines[index] for index in selected]
    if len(result) != args.n_attributes:
        raise RuntimeError(f"concept-name count {len(result)} != {args.n_attributes}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koh-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--class-checkpoint", default="")
    ap.add_argument("--kind", choices=["joint", "two_stage"], required=True)
    ap.add_argument("--data-pkl", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--n-attributes", required=True, type=int)
    ap.add_argument("--names", default="")
    ap.add_argument("--selection-indices", default="")
    ap.add_argument("--attributes", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    curated = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(curated / "compat"))
    sys.path.insert(0, str(Path(args.koh_root).resolve()))
    from CUB.dataset import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    class_model = load_model(args.class_checkpoint, device) if args.class_checkpoint else None
    if args.kind == "two_stage" and class_model is None:
        raise RuntimeError("two_stage export requires --class-checkpoint")
    concept_names = names(args)
    records = pickle.loads(Path(args.data_pkl).read_bytes())
    loader = load_data([args.data_pkl], use_attr=True, no_img=False, batch_size=64,
                       uncertain_label=False, n_class_attr=2, image_dir="images",
                       resampling=False)
    rows, offset = [], 0
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(args.work_dir)
        with torch.inference_mode():
            for images, labels, attribute_labels in loader:
                images = images.to(device)
                outputs = model(images)
                if args.kind == "joint":
                    class_logits, concept_outputs = outputs[0], outputs[1:]
                else:
                    concept_outputs = outputs
                    z_temp = torch.cat([value.reshape(value.shape[0], -1)
                                        for value in concept_outputs], dim=1)
                    class_logits = class_model(torch.sigmoid(z_temp))
                z = torch.cat([value.reshape(value.shape[0], -1)
                               for value in concept_outputs], dim=1)
                if z.shape[1] != args.n_attributes:
                    raise RuntimeError(f"checkpoint emitted {z.shape[1]} concepts")
                probabilities = torch.sigmoid(z)
                predictions = class_logits.argmax(1)
                gt = torch.stack(list(attribute_labels), dim=1) if isinstance(
                    attribute_labels, (list, tuple)) else attribute_labels
                for batch_index in range(images.shape[0]):
                    record = records[offset + batch_index]
                    image_name = str(record.get("image", record["img_path"]))
                    for concept_index, concept_name in enumerate(concept_names):
                        raw = float(z[batch_index, concept_index].cpu())
                        rows.append({
                            "image": image_name,
                            "y_true": int(labels[batch_index]),
                            "y_pred": int(predictions[batch_index]),
                            "concept_index": concept_index,
                            "concept_name": concept_name,
                            "z": raw,
                            "prob": float(probabilities[batch_index, concept_index].cpu()),
                            "gt_label": int(gt[batch_index, concept_index]),
                        })
                offset += images.shape[0]
    finally:
        import os
        os.chdir(old_cwd)
    if offset != len(records):
        raise RuntimeError(f"exported {offset} images but pickle contains {len(records)}")
    frame = pd.DataFrame(rows)
    expected_rows = len(records) * args.n_attributes
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"export has {len(frame)} rows, expected {expected_rows}"
        )
    numeric = frame[["z", "prob"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("export contains non-finite raw logits or probabilities")
    if not frame["prob"].between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError("export contains probabilities outside [0, 1]")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"[KOH EVAL SUCCESS] {offset} images x {args.n_attributes} concepts -> {out}")


if __name__ == "__main__":
    main()
