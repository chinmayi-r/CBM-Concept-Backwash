#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.cub_dataset import CUBDataset
from models.resnet_cub import get_resnet50_cub, load_checkpoint


def make_eval_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


#def get_feature_layers(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    
#    Choose which layers to probe. You can adjust this list.
    
#    return {
#        "conv1": model.conv1,
#        "layer1": model.layer1,
#        "layer2": model.layer2,
#        "layer3": model.layer3,
#        "layer4": model.layer4,
#        "avgpool": model.avgpool,
#    }"""

def get_feature_layers(model: torch.nn.Module) -> Dict[str, torch.nn.Module]: 
    """
    Fine-grained probe points:
      conv1 (after maxpool), each Bottleneck block output, and avgpool.
    """
    layers = {}

    # conv1: use maxpool output (better match to your earlier "stem" probe point)
    layers["conv1"] = model.maxpool

    # each residual block output
    for stage in ["layer1", "layer2", "layer3", "layer4"]:
        seq = getattr(model, stage)  # nn.Sequential
        for b, block in enumerate(seq):
            layers[f"{stage}.{b}"] = block

    layers["avgpool"] = model.avgpool
    return layers



def extract_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    split_name: str,
):
    feature_layers = get_feature_layers(model)
    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in feature_layers}
    labels_list: List[torch.Tensor] = []
    image_ids_list: List[torch.Tensor] = []

    hooks = []

#"""    def make_hook(layer_name: str):
#        def hook(module, inp, out):
#            # out: (B, C, H, W) or (B, C, 1, 1)
#            if out.dim() == 4:
#                feat = out.mean(dim=[2, 3])  # global average pool
#            else:
#                feat = out.view(out.size(0), -1)
#            activations[layer_name].append(feat.detach().cpu())
#        return hook"""

    def make_hook(layer_name: str):
        def hook(module, inp, out):
            # out: (B,C,H,W) or (B,C) or (B,C,1,1)
            if out.dim() == 4:
                feat = out.mean(dim=(2, 3))  # GAP -> (B,C)
            elif out.dim() == 2:
                feat = out
            else:
                feat = out.view(out.size(0), -1)
            activations[layer_name].append(feat.detach().cpu())
        return hook


    for name, module in feature_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {split_name}"):
            imgs = batch["image"].to(device)
            labels = batch["label"]
            image_ids = batch["image_id"]

            _ = model(imgs)

            labels_list.append(labels.cpu())
            image_ids_list.append(image_ids.cpu())

    # remove hooks
    for h in hooks:
        h.remove()

    out_dir.mkdir(parents=True, exist_ok=True)

    labels_tensor = torch.cat(labels_list, dim=0)
    image_ids_tensor = torch.cat(image_ids_list, dim=0)
    torch.save({"labels": labels_tensor, "image_ids": image_ids_tensor}, out_dir / f"labels_{split_name}.pt")

    for name, tensor_list in activations.items():
        feats = torch.cat(tensor_list, dim=0)
        torch.save(feats, out_dir / f"{name}_{split_name}.pt")
        print(f"[extract_features] Saved {name}_{split_name}.pt with shape {feats.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/resnet50_cub_best.pth")
    parser.add_argument("--out_dir", type=str, default="features/resnet50_cub")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[extract_features] Using device:", device)

    tf_eval = make_eval_transform()
    train_ds = CUBDataset(args.cub_root, split="train", transform=tf_eval)
    test_ds = CUBDataset(args.cub_root, split="test", transform=tf_eval)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    model = get_resnet50_cub()
    model = load_checkpoint(model, args.ckpt_path, device=device).to(device)

    out_dir = Path(args.out_dir)
    extract_split(model, train_loader, device, out_dir, "train")
    extract_split(model, test_loader, device, out_dir, "test")


if __name__ == "__main__":
    main()
