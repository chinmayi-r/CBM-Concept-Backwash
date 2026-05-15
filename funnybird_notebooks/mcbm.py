#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ── CELL 0-1: Imports ────────────────────────────────────────────────────────
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 60)
print("imports OK")


# In[2]:


# ── CELL 0-2: Paths ──────────────────────────────────────────────────────────
ROOT    = Path("/scratch/network/cr7998/cv_emergence_project")
FB_ROOT = ROOT / "data" / "FunnyBirds"   # adjust if nested

# MCBM_FEATS keys must match the GAMMAS list defined in Section 6.
# Defined here as a reference dict; missing dirs are filtered out at sweep time.
_ALL_GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
MCBM_FEATS  = {g: ROOT / "features" / f"resnet50_funnybirds_mcbm_gamma{g}"
               for g in _ALL_GAMMAS}

assert FB_ROOT.exists(), f"Missing: {FB_ROOT}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
for g, p in MCBM_FEATS.items():
    print(f"  gamma={g}: exists={p.exists()}  ({p.name})")


# In[3]:


# ── CELL 0-3: Global constants — defined once, referenced everywhere ──────────
LAYERS = [
    "conv1",
    "layer1.0", "layer1.1", "layer1.2",
    "layer2.0", "layer2.1", "layer2.2", "layer2.3",
    "layer3.0", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5",
    "layer4.0", "layer4.1", "layer4.2",
    "avgpool",
]
EMERGE_METRIC = "plain"   # "plain" | "balanced"
EMERGE_FRAC   = 0.90

print(f"EMERGE_METRIC : {EMERGE_METRIC}")
print(f"EMERGE_FRAC   : {EMERGE_FRAC}")
print(f"Layers        : {len(LAYERS)}  ({LAYERS[0]} → {LAYERS[-1]})")


# In[4]:


# ── CELL 1-1: Discover & load FunnyBirds ground truth ────────────────────────
PART_NAMES = ["beak", "tail", "wing", "feet", "eye"]   # extend if needed

def find_gt_json(fb_root: Path) -> Path:
    for name in ["data_gt.json", "gt.json", "annotations.json"]:
        if (fb_root / name).exists():
            return fb_root / name
    hits = sorted(fb_root.rglob("data_gt.json"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No GT JSON found under {fb_root}")

def _parse_record(rec: dict, idx: int) -> dict:
    img_file = rec.get("filename") or rec.get("img_name") or str(idx)
    class_id = (rec["class"] if rec.get("class") is not None
                else rec.get("class_idx", -1))
    split = rec.get("split", "unknown")
    if split == "unknown":
        s = str(img_file)
        split = "train" if "train" in s else ("test" if ("test" in s or "val" in s) else "unknown")

    parts = {}
    for pname in PART_NAMES:
        info = rec.get("parts", {}).get(pname, {})
        if isinstance(info, dict):
            for k, v in info.items():
                parts[f"{pname}.{k}"] = str(v)
        elif info is not None:
            parts[f"{pname}.value"] = str(info)
    bg = rec.get("background") or rec.get("bg")
    if bg is not None:
        parts["background.type"] = str(bg)

    return {"image_id": idx, "filename": img_file,
            "class_id": int(class_id), "split": split, **parts}

gt_path = find_gt_json(FB_ROOT)
print("GT:", gt_path)
with gt_path.open() as f:
    raw_gt = json.load(f)

if isinstance(raw_gt, list):
    records = raw_gt
elif isinstance(raw_gt, dict):
    first_v = next(iter(raw_gt.values()))
    if isinstance(first_v, dict):
        records = [{"filename": k, **v} for k, v in raw_gt.items()]
    else:
        records = [item for v in raw_gt.values()
                   for item in (v if isinstance(v, list) else [v])]
else:
    records = []

parsed  = [_parse_record(r, i) for i, r in enumerate(records)]
raw_df  = pd.DataFrame(parsed)
print(f"Parsed {len(raw_df)} records  |  columns: {list(raw_df.columns)}")
print("Splits:", raw_df["split"].value_counts().to_dict())
raw_df.head(3)


# In[ ]:


# ── CELL 1-2: img_df, class_part_df, class_wide ──────────────────────────────
FIXED_COLS    = {"image_id", "filename", "class_id", "split"}
PART_ATTR_COLS = [c for c in raw_df.columns if c not in FIXED_COLS]

img_df = raw_df[["image_id", "filename", "class_id", "split"]].copy()

# Deterministic per-class part table (one row per class × part_attr)
class_part_rows = []
for col in PART_ATTR_COLS:
    for cls_id, val in raw_df.groupby("class_id")[col].agg(
            lambda x: x.mode().iloc[0]).items():
        part = col.split(".")[0]
        attr = col.split(".")[1] if "." in col else "value"
        class_part_rows.append({"class_id": int(cls_id), "part_attr": col,
                                  "part": part, "attribute": attr, "value": val})

class_part_df = pd.DataFrame(class_part_rows)
class_wide    = class_part_df.pivot(index="class_id", columns="part_attr", values="value")

print(f"img_df       : {len(img_df)} images, {img_df['class_id'].nunique()} classes")
print(f"PART_ATTR_COLS: {PART_ATTR_COLS}")
print(f"class_wide   : {class_wide.shape}")
display(class_wide.head(5))


# In[5]:


# ── CELL 1-3: Full attribute vocabulary ──────────────────────────────────────
attr_records = []
for col in PART_ATTR_COLS:
    for val in sorted(raw_df[col].dropna().unique()):
        n_cls  = int((class_wide[col] == val).sum())
        prev   = n_cls / len(class_wide)
        attr_records.append({
            "attr_name"       : f"{col}::{val}",
            "part_attr"       : col,
            "value"           : val,
            "n_classes"       : n_cls,
            "global_prevalence": float(prev),
        })

attr_df = pd.DataFrame(attr_records)
print(f"Total attributes: {len(attr_df)}")
display(attr_df)


# In[ ]:




