"""Score new CUB pilot edits with an accepted frozen Koh Joint checkpoint.

Review is a separate blinded artifact. This program never marks edits accepted,
never changes model weights, and refuses wrong-framework or legacy artifacts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("edits", "manifest", "checkpoint", "evaluation", "data-pkl", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from canonical_manifest import sha256, verify_manifest
    for p in (args.checkpoint, args.evaluation, args.manifest):
        if "legacy_not_for_notebooks" in str(p.resolve()):
            raise ValueError("Legacy artifacts cannot enter Notebook 05")
    manifest = json.loads(args.manifest.read_text())
    meta = manifest.get("metadata", {})
    if meta.get("framework") != "koh_joint":
        raise ValueError("Official Koh Joint manifest required")
    if meta.get("dataset") != "cub70" or str(meta.get("seed")) != "1":
        raise ValueError("This pilot is declared for CUB70 seed 1 only")
    verify_manifest(argparse.Namespace(manifest=args.manifest))
    recorded = {**manifest.get("inputs",{}), **manifest.get("outputs",{})}
    for p in (args.checkpoint, args.evaluation):
        if recorded.get(str(p.resolve())) != sha256(p):
            raise ValueError(f"Selected artifact not bound to manifest: {p}")
    import torch
    from cub_koh_spatial_audit import load_model, koh_eval_loader_kwargs, koh_pkl_paths, summarize_replay
    from cub_mask_swap_v2 import image_hash
    from cub_intervention_analysis import REVIEWS
    from CUB.dataset import load_data
    if "train.pkl" in str(args.data_pkl):
        raise ValueError("Frozen pilot requires the accepted evaluation pickle, not an augmented train loader")
    loader = load_data(koh_pkl_paths(args.data_pkl), **koh_eval_loader_kwargs())
    transform = loader.dataset.transform
    model = load_model(args.checkpoint, torch.device(args.device))
    if model.sec_model.linear.in_features != 112 or model.sec_model.linear.out_features != 70:
        raise ValueError("Expected CUB70 112-concept, 70-species model")
    evaluation = pd.read_parquet(args.evaluation)
    evaluation["stem"] = evaluation.image.map(lambda p: Path(str(p)).stem)
    if evaluation.duplicated(["stem","concept_index"]).any():
        raise ValueError("Ambiguous evaluation image identities")
    rows, replay_records = [], []
    edit_files = sorted(args.edits.glob("*/EDIT.json"))
    if not edit_files:
        raise ValueError("No generated edit manifests")
    with torch.inference_mode():
        for path in edit_files:
            edit = json.loads(path.read_text()); trial = edit["trial"]
            expected = evaluation[evaluation.stem == Path(trial["source_image"]).stem].sort_values("concept_index")
            if len(expected) != 112 or expected.concept_index.tolist() != list(range(112)):
                raise ValueError("Source image must have 112 accepted evaluation concepts")
            indices = {}
            for role in ("source","donor"):
                candidates = expected[expected.concept_name == trial[role+"_concept"]]
                if len(candidates) != 1:
                    raise ValueError("Exact concept name absent or ambiguous")
                indices[role] = int(candidates.iloc[0].concept_index)
            row = {k:trial[k] for k in ("trial_id","attribute_type","source_concept","donor_concept","mask_group")}
            row.update(source_image=trial["source_image"], same_value_donor_image=trial["same_value_image"],
                different_value_donor_image=trial["different_value_image"], seed=int(meta["seed"]),
                **{k:False for k in REVIEWS})
            for variant in ("original","self","same_value","different_value"):
                image = Image.open(path.parent/(variant+".png")).convert("RGB")
                pixels = image_hash(np.asarray(image))
                expected_hash = (edit["edits"]["self"]["original_rgb_sha256"] if variant == "original"
                                 else edit["edits"][variant]["edited_rgb_sha256"])
                if pixels != expected_hash:
                    raise ValueError("Edited pixels differ from recorded manifest")
                result = model(transform(image).unsqueeze(0).to(args.device))
                if not isinstance(result,(list,tuple)) or len(result) != 113:
                    raise ValueError("Unexpected official Koh output contract")
                z = torch.cat(result[1:],dim=1).cpu().numpy()
                if not np.isfinite(z).all():
                    raise ValueError("Non-finite raw concept logits")
                if variant == "original":
                    report = summarize_replay(expected.z.to_numpy()[None,:], z,
                        np.array([expected.iloc[0].y_pred]), result[0].argmax(1).cpu().numpy())
                    report["trial_id"] = trial["trial_id"]; replay_records.append(report)
                    if not report["accepted"]:
                        raise ValueError("Original source prediction does not replay accepted export")
                for role in ("source","donor"):
                    row[f"z_{role}_{variant}"] = float(z[0,indices[role]])
                prefix = "source" if variant == "original" else variant
                row[prefix+"_pixels_sha256"] = pixels
            for role in ("source","donor"):
                if row[f"z_{role}_self"] != row[f"z_{role}_original"]:
                    raise ValueError("Identical self reconstruction did not reproduce same-session logits")
            rows.append(row)
            print(f"Frozen inference {len(rows)}/{len(edit_files)}: {trial['trial_id']}; visual review remains separate",flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_parquet(args.out/"predictions_unreviewed.parquet",index=False)
    # Blinded template contains no logits. Review image grids before joining it.
    pd.DataFrame(rows)[["trial_id","seed"]+REVIEWS].to_csv(args.out/"blinded_review.csv",index=False)
    (args.out/"INFERENCE.json").write_text(json.dumps(dict(status="METHOD NOT CALIBRATED",
        framework="koh_joint", training=False, checkpoint_sha256=sha256(args.checkpoint),
        source_manifest=str(args.manifest.resolve()), original_replay=replay_records,
        predictions_sha256=sha256(args.out/"predictions_unreviewed.parquet")),indent=2)+"\n")


if __name__ == "__main__": main()
