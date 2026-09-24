"""Paper-inspired offline CLIP -> GroundingDINO -> SAM region extraction.

Uses local weight snapshots only. No training or automatic weight downloads.
Released CUB masks remain the primary source on covered images. This creates
region proposals, not verified absence labels or accepted causal interventions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def select_and_merge(masks, labels, min_pixels, max_fraction=.5, merge_iou=.5):
    """Deterministic greedy union; order is detector confidence order.

    A merged region retains every contributing text label. It must never be
    silently interpreted as a uniquely identified anatomical part.
    """
    if min_pixels < 1 or not 0 < max_fraction <= 1 or not 0 < merge_iou <= 1:
        raise ValueError("Invalid region thresholds")
    if len(masks) != len(labels):
        raise ValueError("Mask/label count mismatch")
    regions = []
    for mask, label in zip(masks, labels):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("Masks must be nonempty 2D arrays")
        if regions and mask.shape != regions[0]["mask"].shape:
            raise ValueError("Mask grids differ")
        if mask.sum() < min_pixels or mask.mean() > max_fraction:
            continue
        found = None
        for region in regions:
            union = region["mask"] | mask
            overlap = (region["mask"] & mask).sum() / union.sum()
            if overlap >= merge_iou and union.mean() <= max_fraction:
                found = region
                break
        if found is None:
            regions.append({"mask": mask.copy(), "labels": [str(label)]})
        else:
            found["mask"] |= mask
            if str(label) not in found["labels"]:
                found["labels"].append(str(label))
    return regions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "vocabulary", "clip", "detector", "sam", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--min-pixels", type=int, required=True,
                    help="Declared reconstruction choice; paper does not give a numeric value")
    ap.add_argument("--box-threshold", type=float, default=.25)
    ap.add_argument("--text-threshold", type=float, default=.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    paths = json.loads(args.images.read_text())
    vocab = json.loads(args.vocabulary.read_text())
    if not paths or not vocab or len(set(vocab)) != len(vocab):
        raise ValueError("Nonempty image list and unique text vocabulary required")
    if args.min_pixels < 1:
        raise ValueError("min-pixels must be positive")
    for path in [*map(Path, paths), args.clip, args.detector, args.sam]:
        if not path.exists():
            raise FileNotFoundError(path)
    import torch
    from transformers import (CLIPModel, CLIPProcessor, AutoProcessor,
                              AutoModelForZeroShotObjectDetection, SamModel, SamProcessor)
    device = torch.device(args.device)
    clip = CLIPModel.from_pretrained(args.clip, local_files_only=True).to(device).eval()
    if clip.config.vision_config.patch_size != 16 or clip.config.vision_config.hidden_size != 768:
        raise ValueError("Declared paper lane requires CLIP ViT-B/16")
    cp = CLIPProcessor.from_pretrained(args.clip, local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector, local_files_only=True).to(device).eval()
    dp = AutoProcessor.from_pretrained(args.detector, local_files_only=True)
    sam = SamModel.from_pretrained(args.sam, local_files_only=True).to(device).eval()
    sp = SamProcessor.from_pretrained(args.sam, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.inference_mode():
        for i, path in enumerate(map(Path, paths)):
            image = Image.open(path).convert("RGB")
            inputs = cp(text=vocab, images=image, return_tensors="pt", padding=True).to(device)
            ranked = clip(**inputs).logits_per_image[0].argsort(descending=True)[:32].tolist()
            selected = [vocab[k] for k in ranked]
            prompt = ". ".join(selected) + "."
            din = dp(images=image, text=prompt, return_tensors="pt").to(device)
            detected = dp.post_process_grounded_object_detection(
                detector(**din), din.input_ids, threshold=args.box_threshold,
                text_threshold=args.text_threshold, target_sizes=[image.size[::-1]])[0]
            order = detected["scores"].argsort(descending=True)
            boxes = detected["boxes"][order].cpu().tolist()
            labels = [detected["text_labels"][k] for k in order.tolist()]
            regions = []
            if boxes:
                sin = sp(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
                sout = sam(**sin, multimask_output=False)
                masks = sp.image_processor.post_process_masks(sout.pred_masks.cpu(),
                    sin["original_sizes"].cpu(), sin["reshaped_input_sizes"].cpu())[0][:,0].numpy()
                regions = select_and_merge(masks, labels, args.min_pixels)
            folder = args.out / f"image_{i:05d}"
            folder.mkdir()
            for j, region in enumerate(regions):
                Image.fromarray(region["mask"].astype(np.uint8)*255).save(folder/f"region_{j:03d}.png")
            records.append(dict(image=str(path.resolve()), image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                folder=folder.name, ranked_concepts=selected, boxes=boxes, detection_labels=labels,
                regions=[dict(labels=r["labels"], pixels=int(r["mask"].sum())) for r in regions],
                absence_verified=False))
            print(f"Region extraction {i+1}/{len(paths)}: {len(regions)} proposals; no absence inference", flush=True)
    manifest = dict(status="METHOD NOT CALIBRATED", training=False,
        paper="https://arxiv.org/abs/2510.04180v1", settings_source="v2 Appendix K",
        top_k=32, maximum_area_fraction=.5, merge_iou=.5,
        reconstruction_choices=dict(min_pixels=args.min_pixels, box_threshold=args.box_threshold,
            text_threshold=args.text_threshold, merging="confidence-ordered greedy union", sam_multimask=False),
        snapshots={k:str(getattr(args,k).resolve()) for k in ("clip","detector","sam")},
        images=records, vocabulary=vocab)
    (args.out/"SEGMENTATION.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
