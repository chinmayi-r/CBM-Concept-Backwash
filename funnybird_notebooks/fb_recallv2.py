#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


LAYERS = [
    "conv1",
    "layer1.0", "layer1.1", "layer1.2",
    "layer2.0", "layer2.1", "layer2.2", "layer2.3",
    "layer3.0", "layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5",
    "layer4.0", "layer4.1", "layer4.2",
    "avgpool",
]

EMERGE_METRIC = "balanced"   # "balanced" | "plain"
EMERGE_FRAC   = 0.90         # e.g. 0.90 | 0.95

# ── Choose emergence criterion + tag pre / at / post ──────────────────────────
# EMERGE_CRITERION reads emerge_idx_frac (uses EMERGE_FRAC) or emerge_idx_jump
# Both were computed by attribute_emergence() using EMERGE_METRIC and EMERGE_FRAC above.
EMERGE_CRITERION = "sharp_jump"    # "frac90" | "sharp_jump"
emerge_col = "emerge_idx_frac"   if EMERGE_CRITERION == "frac90" else "emerge_idx_jump"
layer_col  = "emerge_layer_frac" if EMERGE_CRITERION == "frac90" else "emerge_layer_jump"


# In[3]:


import sys
ROOT = Path('/scratch/network/cr7998/cv_emergence_project')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FB         = ROOT / 'data'     / 'FunnyBirds'
BASE_FEATS = ROOT / 'features' / 'resnet50_funnybirds'
CBM_FEATS  = ROOT / 'features' / 'resnet50_cbm_funnybirds'

assert FB.exists(),                           f'Missing FunnyBirds folder: {FB}'
assert (FB / 'dataset_train.json').exists(),  f'Missing dataset_train.json: {FB}'

for name, d in [('baseline', BASE_FEATS), ('cbm', CBM_FEATS)]:
    if not d.exists(): print(f'  [warn] {name} features not extracted: {d}')
    else:              print(f'  [ok]   {name}: {d}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[config] device: {device}')


# In[4]:


def load_species_maps(fb_root: Path):
    """
    Load species ID→name mappings from FunnyBirds metadata/classes.csv.
    Mirrors load_species_maps(cub_root) in recall.ipynb.
    """
    classes_csv = fb_root / "metadata" / "classes.csv"
    if not classes_csv.exists():
        raise FileNotFoundError(
            "metadata/classes.csv not found. Run prepare_funnybirds_metadata.py first."
        )
    df = pd.read_csv(classes_csv)
    id2name  = dict(zip(df["class_id"], df["class_name"]))
    id2short = {k: v.replace("funnybird_", "FB") for k, v in id2name.items()}
    return id2name, id2short


def load_meta(fb_root: Path) -> pd.DataFrame:
    """
    Load per-image metadata with species information.
    Returns DataFrame with columns: image_id, class_id, species_id, species_name, is_train.
    Mirrors load_meta(cub_root) in recall.ipynb.
    """
    images_csv = fb_root / "metadata" / "images.csv"
    if not images_csv.exists():
        raise FileNotFoundError(
            "metadata/images.csv not found. Run prepare_funnybirds_metadata.py first."
        )
    df = pd.read_csv(images_csv)
    id2name, _ = load_species_maps(fb_root)
    df["species_id"]   = df["class_id"]
    df["species_name"] = df["class_id"].map(id2name)
    return df


def load_image_attr_labels_robust(fb_root: Path) -> pd.DataFrame:
    """
    Load per-image binary concept labels from metadata/image_concepts_binary.csv.
    Mirrors load_image_attr_labels_robust(cub_root) in recall.ipynb.

    Returns long-form DataFrame: image_id, attr_id, attr_name, is_present, certainty.
    certainty is always 1 for FunnyBirds (ground-truth, no annotation noise).
    """
    concepts_csv = fb_root / "metadata" / "image_concepts_binary.csv"
    if not concepts_csv.exists():
        raise FileNotFoundError(
            "metadata/image_concepts_binary.csv not found. "
            "Run prepare_funnybirds_metadata.py first."
        )
    wide = pd.read_csv(concepts_csv)
    concept_cols = [c for c in wide.columns if c != "image_id"]
    long = wide.melt(
        id_vars="image_id", value_vars=concept_cols,
        var_name="attr_name", value_name="is_present"
    )
    long["attr_id"]    = long.groupby("attr_name", sort=False).ngroup()
    long["is_present"] = long["is_present"].astype(int)
    long["certainty"]  = 1   # FunnyBirds: ground-truth, always certain
    return long[["image_id", "attr_id", "attr_name", "is_present", "certainty"]]


def load_attr_maps(fb_root: Path):
    """
    Load concept names from metadata/concepts.csv.
    Mirrors load_attr_maps(attr_txt) in recall.ipynb.
    Returns: attr_id_to_name, attr_name_to_id
    """
    concepts_csv = fb_root / "metadata" / "concepts.csv"
    if not concepts_csv.exists():
        raise FileNotFoundError(
            "metadata/concepts.csv not found. Run prepare_funnybirds_metadata.py first."
        )
    df = pd.read_csv(concepts_csv)
    id2name  = dict(zip(df["concept_id"], df["concept_name"]))
    name2id  = dict(zip(df["concept_name"], df["concept_id"]))
    return id2name, name2id


# In[5]:


# FunnyBirds: all 26 concepts (no prevalence filter needed — one-hot per part group)
from datasets.funnybirds_dataset import concept_names as _fb_concept_names
ATTR_LIST = _fb_concept_names()
print(f"FunnyBirds concepts ({len(ATTR_LIST)}): {ATTR_LIST}")


# In[6]:


# Class-concept matrix: every row = species, every column = one of 26 part-variant concepts.
# EXACT ground truth — no annotation noise. Key FunnyBirds advantage.
from datasets.funnybirds_dataset import FunnyBirdsDataset
_fb_ds = FunnyBirdsDataset(FB, split="train")
class_concept_matrix, _ = _fb_ds.get_class_concept_matrix()

cc_df = pd.DataFrame(
    class_concept_matrix.numpy(),
    columns=_fb_concept_names(),
    index=[f"funnybird_{i:02d}" for i in range(class_concept_matrix.shape[0])],
)
print(f"Class-concept matrix shape: {cc_df.shape}")
print("Each row should sum to 5 (one variant per part per species):")
print(cc_df.sum(axis=1).value_counts())
print("\nFirst 5 species:")
cc_df.head()


# In[7]:


# Load metadata, concept labels, and attribute maps
meta          = load_meta(FB)
img_attr_long = load_image_attr_labels_robust(FB)
attr_id_to_name, attr_name_to_id = load_attr_maps(FB)

# attr_df: one row per concept — mirrors CUB attr_df used downstream
attr_df = pd.DataFrame({
    "attr_name": list(attr_name_to_id.keys()),
    "attr_id":   list(attr_name_to_id.values()),
})

# spname: species id → display name (used throughout matched-pair eval)
id2name, _ = load_species_maps(FB)
def spname(sid: int) -> str:
    return id2name.get(int(sid), f"funnybird_{int(sid):02d}")

print(f"meta: {len(meta)} images  "
      f"({meta['is_train'].sum()} train, {(meta['is_train']==0).sum()} test)")
print(f"img_attr_long: {len(img_attr_long)} rows, "
      f"{img_attr_long['attr_name'].nunique()} concepts")
print(f"attr_df: {len(attr_df)} concepts")
print(f"Concepts: {list(attr_name_to_id.keys())}")


# In[8]:


# What this cell does:
# - For a given attr_id, merges meta (image->species, split) with attribute labels (image->y)
# - Produces a clean table with y in {0,1}
# Why it matters:
# - This is the ground-truth label table used for training and evaluation.

def build_attr_labeled_df(
    meta: pd.DataFrame,
    img_attr_long: pd.DataFrame,
    attr_id: int,
    min_certainty: int = 1,
) -> pd.DataFrame:
    """
    Returns dataframe with:
      image_id, species_id, species_name, is_train, y, certainty
    Only keeps annotations with certainty >= min_certainty.
    For FunnyBirds certainty is always 1, so the filter is a no-op.
    """
    sub = img_attr_long[img_attr_long["attr_id"] == int(attr_id)].copy()
    sub = sub[sub["certainty"] >= int(min_certainty)].copy()
    out = meta.merge(sub[["image_id", "is_present", "certainty"]], on="image_id", how="inner")
    out = out.rename(columns={"is_present": "y"})
    out["y"] = out["y"].astype(int)
    return out[["image_id", "species_id", "species_name", "is_train", "y", "certainty"]]


print("Defined: build_attr_labeled_df")

# Sanity check on one concept
attr_name_check = "beak_0"
aid_check = attr_name_to_id[attr_name_check]
lab_check = build_attr_labeled_df(meta, img_attr_long, aid_check, min_certainty=1)
print(f"Attribute: {attr_name_check}  rows labeled: {len(lab_check)}  "
      f"pos rate: {lab_check.y.mean():.4f}")
lab_check.head()


# In[9]:


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_features(feat_dir: Path, layer: str, split: str) -> torch.Tensor:
    p = feat_dir / f"{layer}_{split}.pt"
    assert p.exists(), f"Missing: {p}"
    X = safe_torch_load(p)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    return X.float()


def to_1d_int_array(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x).reshape(-1).astype(int)


def infer_kind(arr):
    if arr.max() <= 200 and arr.min() >= 0:
        return "species_id_like"
    if arr.max() > 200:
        return "image_id_like"
    return "unknown"


def load_split_order(feat_dir, split):
    p = feat_dir / f"labels_{split}.pt"
    assert p.exists(), f"Missing: {p}"
    t = safe_torch_load(p)
    assert isinstance(t, dict), f"Expected dict in {p}, got {type(t)}"
    assert "image_ids" in t, f"{p} missing 'image_ids' key; has {list(t.keys())}"
    ids  = to_1d_int_array(t["image_ids"])
    kind = infer_kind(ids)
    return kind, ids


# LAYER is NOT set here — computed dynamically from species emergence curve below
LAYER = None

base_kind_tr, base_ids_tr = load_split_order(BASE_FEATS, "train") if BASE_FEATS.exists() else (None, None)
base_kind_te, base_ids_te = load_split_order(BASE_FEATS, "test")  if BASE_FEATS.exists() else (None, None)
cbm_kind_tr,  cbm_ids_tr  = load_split_order(CBM_FEATS,  "train") if CBM_FEATS.exists()  else (None, None)
cbm_kind_te,  cbm_ids_te  = load_split_order(CBM_FEATS,  "test")  if CBM_FEATS.exists()  else (None, None)

print("BASE_FEATS available:", BASE_FEATS.exists())
print("CBM_FEATS  available:", CBM_FEATS.exists())
if BASE_FEATS.exists():
    print(f"  Baseline train: {base_kind_tr}  id range: ({base_ids_tr.min()}, {base_ids_tr.max()})")
    print(f"  Baseline test:  {base_kind_te}  id range: ({base_ids_te.min()}, {base_ids_te.max()})")


# In[10]:


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def frac_of_final_idx(vals: np.ndarray, frac: float = 0.90) -> int:
    """Earliest index i such that vals[i] >= frac * vals[-1]."""
    v = pd.Series(np.asarray(vals, dtype=float)).ffill().bfill().to_numpy()
    target = frac * float(v[-1])
    for i, x in enumerate(v):
        if float(x) >= target:
            return int(i)
    return int(len(v) - 1)


def sharp_rise_idx(vals: np.ndarray) -> int:
    """Index of the largest single-step increase in vals."""
    v = pd.Series(np.asarray(vals, dtype=float)).ffill().bfill().to_numpy()
    diffs = np.diff(v)
    return int(np.argmax(diffs)) + 1


def train_linear_probe_multiclass(
    Xtr, ytr, Xte, yte, *, epochs=6, lr=3e-3, wd=1e-4, seed=0, device=None,
):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long,    device=device)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    d, C  = Xtr_t.shape[1], int(ytr_t.max().item()) + 1
    model = nn.Linear(d, C).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(dim=1).detach().cpu().numpy()
    return float((pred == yte).mean())


def train_linear_probe_binary_weighted(
    Xtr, ytr, Xte, yte, *,
    epochs=8, lr=3e-3, wd=1e-4, seed=0, threshold=0.5, device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    ytr_np = np.asarray(ytr, dtype=np.int32).reshape(-1)
    yte_np = np.asarray(yte, dtype=np.int32).reshape(-1)
    Xtr_t  = torch.as_tensor(Xtr, dtype=torch.float32).to(device)
    Xte_t  = torch.as_tensor(Xte, dtype=torch.float32).to(device)
    ytr_t  = torch.as_tensor(ytr_np, dtype=torch.float32).view(-1, 1).to(device)
    d      = int(Xtr_t.shape[1])
    model  = nn.Linear(d, 1).to(device)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos    = float(ytr_np.sum())
    neg    = float(len(ytr_np) - ytr_np.sum())
    if pos <= 0:
        probs = np.zeros_like(yte_np, dtype=float)
        pred  = np.zeros_like(yte_np, dtype=int)
        return 0.5, float((pred == yte_np).mean()), 0.0, probs
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).view(-1).cpu().numpy()
    pred = (probs >= threshold).astype(int)
    ba   = balanced_accuracy(yte_np, pred)
    return float(ba), float((pred == yte_np).mean()), float(pred.mean()), probs


print("Defined: balanced_accuracy  frac_of_final_idx  sharp_rise_idx")
print("Defined: train_linear_probe_multiclass  train_linear_probe_binary_weighted")


# In[11]:


import gc

# ── Alignment indices ──────────────────────────────────────────────────────────
base_ids_tr_int = np.asarray(base_ids_tr, dtype=int)
base_ids_te_int = np.asarray(base_ids_te, dtype=int)

meta_tr_set = set(meta[meta["is_train"] == 1]["image_id"].astype(int))
meta_te_set = set(meta[meta["is_train"] == 0]["image_id"].astype(int))

keep_tr_idx = np.array([i for i, iid in enumerate(base_ids_tr_int) if iid in meta_tr_set], dtype=int)
keep_te_idx = np.array([i for i, iid in enumerate(base_ids_te_int) if iid in meta_te_set], dtype=int)

aligned_tr_ids = base_ids_tr_int[keep_tr_idx]
aligned_te_ids = base_ids_te_int[keep_te_idx]

# ── Load all layers — slice immediately, never keep full unindexed tensors ─────
aligned_base_tr = {L: load_features(BASE_FEATS, L, "train")[keep_tr_idx] for L in LAYERS}
aligned_base_te = {L: load_features(BASE_FEATS, L, "test")[keep_te_idx]  for L in LAYERS}
gc.collect()

print(f"Loaded {len(LAYERS)} layers.  layer4.0 shape: {aligned_base_tr['layer4.0'].shape}")

# ── Species labels — offset by min species_id so indices are always 0-based ───
_meta_idx = meta.set_index("image_id")
_sid_min  = int(meta["species_id"].min())

ysp_tr_ba = _meta_idx.loc[aligned_tr_ids, "species_id"].to_numpy(dtype=np.int64) - _sid_min
ysp_te_ba = _meta_idx.loc[aligned_te_ids, "species_id"].to_numpy(dtype=np.int64) - _sid_min

assert ysp_tr_ba.min() >= 0, "Negative class index — species_id offset wrong!"
print(f"species_id range: {_sid_min} → {int(meta['species_id'].max())}  (offset={_sid_min})")
print(f"ysp_tr_ba range: {ysp_tr_ba.min()} → {ysp_tr_ba.max()}")

# ── Species emergence curve — computed NOW so LAYER can be set dynamically ─────
# Must run before screening and run_many, both of which need LAYER.
print("\nComputing species emergence curve across layers...")
species_curve_ba = np.array([
    train_linear_probe_multiclass(
        aligned_base_tr[L], ysp_tr_ba,
        aligned_base_te[L], ysp_te_ba,
        epochs=15, seed=0,
    )
    for L in LAYERS
], dtype=float)

species_emerge_idx_jump = sharp_rise_idx(species_curve_ba)
species_emerge_idx_frac = frac_of_final_idx(species_curve_ba, frac=EMERGE_FRAC)

print(f"\nSpecies emergence:")
print(f"  sharp_jump : layer index {species_emerge_idx_jump} = {LAYERS[species_emerge_idx_jump]}"
      f"  (acc={species_curve_ba[species_emerge_idx_jump]:.3f})")
print(f"  frac{int(EMERGE_FRAC*100)}     : layer index {species_emerge_idx_frac} = {LAYERS[species_emerge_idx_frac]}"
      f"  (acc={species_curve_ba[species_emerge_idx_frac]:.3f})")

# ── Set LAYER from the configured criterion ────────────────────────────────────
if EMERGE_CRITERION == "frac90":
    species_emerge_idx_ba = species_emerge_idx_frac
else:
    species_emerge_idx_ba = species_emerge_idx_jump

LAYER = LAYERS[species_emerge_idx_ba]
print(f"\nLAYER set to: {LAYER}  (criterion: {EMERGE_CRITERION})")

assert aligned_base_tr[LAYERS[0]].shape[0] == len(aligned_tr_ids)
assert aligned_base_te[LAYERS[0]].shape[0] == len(aligned_te_ids)
print("Feature / label alignment OK")


# In[12]:


fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(species_curve_ba, marker="o", linewidth=2, label="Species ID accuracy")
ax.axvline(species_emerge_idx_ba, color="red", linestyle="--", linewidth=1.5,
           label=f"{EMERGE_CRITERION}: {LAYER}")
ax.set_xticks(range(len(LAYERS)))
ax.set_xticklabels(LAYERS, rotation=45, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Test accuracy (multiclass)")
ax.set_title(
    f"Species identity emergence — Baseline ResNet (FunnyBirds)\n"
    f"criterion={EMERGE_CRITERION}  EMERGE_FRAC={EMERGE_FRAC}  → LAYER={LAYER}"
)
ax.legend()
plt.tight_layout()
plt.show()


# In[13]:


# What this cell does:
# - Aligns features (in feature row order) to attribute labels by image_id.
# - Produces X_aligned and df_aligned with the same ordering.

def align_features_and_labels(
    X_split: torch.Tensor,
    image_ids_in_feature_order: np.ndarray,
    labeled_df_split: pd.DataFrame,
):
    """
    Inputs:
      X_split: feature tensor [N, D]
      image_ids_in_feature_order: length N
      labeled_df_split: DataFrame with [image_id, y, species_id, species_name]
    Output:
      X_aligned, df_aligned (same ordering, only images that have labels)
    """
    labeled  = labeled_df_split.set_index("image_id")[["y", "species_id", "species_name"]]
    keep_idx = []
    rows     = []
    for i, img_id in enumerate(image_ids_in_feature_order):
        img_id = int(img_id)
        if img_id in labeled.index:
            keep_idx.append(i)
            y, sid, sname = labeled.loc[img_id]
            rows.append((img_id, int(sid), str(sname), int(y)))
    X_aligned  = X_split[keep_idx]
    df_aligned = pd.DataFrame(rows, columns=["image_id", "species_id", "species_name", "y"])
    return X_aligned, df_aligned


# In[14]:


# What this cell does:
# - Defines a linear probe (single linear layer).
# - Trains with BCEWithLogitsLoss + mild class-imbalance handling.
# Why it matters:
# - Probe is the measurement instrument: "is the attribute encoded in features?"

class LinearProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def train_probe(
    Xtr: torch.Tensor, ytr: np.ndarray,
    seed=0, lr=1e-2, wd=1e-4, epochs=25, batch=512,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    Xtr   = Xtr.to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    probe = LinearProbe(Xtr.shape[1]).to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    pos   = float(ytr_t.mean().item())
    pos_weight = (
        torch.tensor([(1 - pos) / pos], device=device)
        if 0 < pos < 1
        else torch.tensor([1.0], device=device)
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = Xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx    = perm[i:i+batch]
            logits = probe(Xtr[idx])
            loss   = loss_fn(logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


@torch.no_grad()
def predict_probs(probe: nn.Module, X: torch.Tensor, batch=4096) -> np.ndarray:
    probe.eval()
    probs = []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i+batch].to(device)
        logits = probe(xb)
        probs.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(probs, dim=0).numpy()


print("Defined: LinearProbe  train_probe  predict_probs")


# In[15]:


# What this cell does:
# - Finds species that have enough positives and negatives for the chosen attribute.
# - Samples many species pairs.
# - For each pair, subsamples to match prevalence exactly and computes recall on positives.
# Why it matters:
# - This isolates species-specific recall differences even when prevalence is controlled.
#
# FunnyBirds note: every attribute is fixed per species (all-positive or all-negative —
# no within-species variation). make_candidate_pairs always returns 0 pairs.
# make_candidate_pairs_fb instead pairs two ALL-POSITIVE species and measures whether
# the probe recalls the attribute equally well for both, giving the same entanglement
# signal without requiring within-species negatives.

def make_candidate_pairs(df_test: pd.DataFrame, min_each=10, max_pairs=200, seed=0):
    """
    Returns list of (sid_A, sid_B, mpos, mneg) 4-tuples.
    For FunnyBirds this usually returns [] — use make_candidate_pairs_fb instead.
    """
    g    = df_test.groupby("species_id")["y"].agg(["count", "sum"]).rename(columns={"sum": "pos"})
    g["neg"] = g["count"] - g["pos"]
    ok   = g[(g["pos"] >= min_each) & (g["neg"] >= min_each)]
    sids = ok.index.to_list()
    rng  = np.random.default_rng(seed)
    pairs = []
    if len(sids) < 2:
        return pairs
    for _ in range(max_pairs * 10):
        a, b = rng.choice(sids, size=2, replace=False)
        mpos = int(min(ok.loc[a, "pos"], ok.loc[b, "pos"]))
        mneg = int(min(ok.loc[a, "neg"], ok.loc[b, "neg"]))
        if mpos >= min_each and mneg >= min_each:
            pairs.append((int(a), int(b), mpos, mneg))
        if len(pairs) >= max_pairs:
            break
    return pairs


def make_candidate_pairs_fb(df_test: pd.DataFrame, min_pos: int = 3,
                             max_pairs: int = 200, seed: int = 0):
    """
    FunnyBirds variant: enumerates ALL pairs of all-positive species
    (prevalence >= 0.9, n_pos >= min_pos).

    Returns list of (sid_A, sid_B, mpos) 3-tuples where
    mpos = min(n_pos_A, n_pos_B).

    Entanglement signal: if the probe fires differently on images from species A vs B
    even though both truly have the attribute → species-identity leakage.
    """
    from itertools import combinations as _combinations
    g    = df_test.groupby("species_id")["y"].agg(["count", "sum"]).rename(columns={"sum": "pos"})
    g["prev"] = g["pos"] / g["count"]
    ok   = g[(g["pos"] >= min_pos) & (g["prev"] >= 0.9)]
    sids = ok.index.tolist()
    pairs = []
    for a, b in _combinations(sids, 2):
        mpos = int(min(ok.loc[a, "pos"], ok.loc[b, "pos"]))
        pairs.append((int(a), int(b), mpos))
        if len(pairs) >= max_pairs:
            break
    return pairs


# In[16]:


def matched_pair_eval(
    df_test: pd.DataFrame, probs: np.ndarray,
    sid_A: int, sid_B: int, mpos: int, mneg: int,
    seed=0, thr=0.5,
):
    """
    CUB-style: subsamples mpos positives + mneg negatives from each species.
    Computes recall on positive examples only.
    (Used when standard pairs exist; for FunnyBirds use fb_pair_eval.)
    """
    df = df_test.copy()
    df["prob"] = probs
    A = df[df.species_id == sid_A]
    B = df[df.species_id == sid_B]
    A_s = pd.concat([A[A.y==1].sample(mpos, random_state=seed),
                     A[A.y==0].sample(mneg, random_state=seed)])
    B_s = pd.concat([B[B.y==1].sample(mpos, random_state=seed),
                     B[B.y==0].sample(mneg, random_state=seed)])

    def recall_pos(d):
        pos  = d[d.y == 1]
        pred = (pos.prob.values >= thr).astype(int)
        return float((pred == 1).mean()) if len(pos) else np.nan

    recA = recall_pos(A_s)
    recB = recall_pos(B_s)
    return {
        "sid_A": sid_A, "sid_B": sid_B,
        "species_A": spname(sid_A), "species_B": spname(sid_B),
        "npos": int(mpos), "nneg": int(mneg),
        "recall_A": float(recA), "recall_B": float(recB),
        "gap": float(abs(recA - recB)),
    }


def fb_pair_eval(
    df_test: pd.DataFrame, probs: np.ndarray,
    sid_A: int, sid_B: int, mpos: int,
    seed: int = 0, thr: float = 0.5,
):
    """
    FunnyBirds variant of matched_pair_eval for all-positive species pairs.

    Samples mpos images (with replacement for bootstrap) from each all-positive species
    and computes recall = fraction classified positive. No negatives are drawn.
    The gap is the absolute difference in recall between species A and B.
    """
    df = df_test.copy()
    df["prob"] = probs
    A = df[(df.species_id == sid_A) & (df.y == 1)]
    B = df[(df.species_id == sid_B) & (df.y == 1)]
    n_A = min(mpos, len(A))
    n_B = min(mpos, len(B))
    #A_s = A.sample(n_A, random_state=seed, replace=(n_A < mpos))
    #B_s = B.sample(n_B, random_state=seed, replace=(n_B < mpos))

    # for abv, replace is only True when you don't have enough images. But for FunnyBirds test set with say 10 images per species and mpos=10, you're sampling all 10 from 10 without replacement — which always returns the same set regardless of seed. The bootstrap loop runs 100 times and picks identical images every time.

    A_s = A.sample(n_A, random_state=seed, replace=True)
    B_s = B.sample(n_B, random_state=seed, replace=True)

    def recall_pos(d):
        pred = (d.prob.values >= thr).astype(int)
        return float(pred.mean()) if len(d) > 0 else np.nan

    recA = recall_pos(A_s)
    recB = recall_pos(B_s)

    # Fixed: recall_pos returns np.nan (not None) on empty — check with np.isnan
    rec_A_valid = not np.isnan(recA)
    rec_B_valid = not np.isnan(recB)
    return {
        "sid_A": sid_A, "sid_B": sid_B,
        "species_A": spname(sid_A), "species_B": spname(sid_B),
        "npos": int(mpos), "nneg": 0,
        "recall_A": float(recA) if rec_A_valid else np.nan,
        "recall_B": float(recB) if rec_B_valid else np.nan,
        "gap": float(abs(recA - recB)) if (rec_A_valid and rec_B_valid) else np.nan,
    }


# In[17]:


def bootstrap_ci(x, alpha=0.05):
    """
    Percentile bootstrap CI for a 1D array x.
    Returns (lo, hi). If x is empty or all-nan, returns (nan, nan).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(x, alpha / 2)),
            float(np.quantile(x, 1 - alpha / 2)))


def bootstrap_p_value(values, null=0.0):
    """
    Two-sided bootstrap p-value for H0: E[value] == null.
    p = 2 * min(P(value <= null), P(value >= null))
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan
    return float(2.0 * min(np.mean(v <= null), np.mean(v >= null)))


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = b != 0
    out[m] = a[m] / b[m]
    return out


# In[18]:


def _build_pair_summary(res_long: pd.DataFrame) -> pd.DataFrame:
    """Shared aggregation logic for both CUB and FunnyBirds bootstrap summaries."""
    def _ci_lo(x): return bootstrap_ci(x)[0]
    def _ci_hi(x): return bootstrap_ci(x)[1]

    pair_summary = (
        res_long.groupby(["sid_A", "sid_B", "species_A", "species_B"], as_index=False)
        .agg(
            npos=(    "npos", "min"),
            nneg=(    "nneg", "min"),
            gap_mean=("gap",  "mean"),
            gap_std=( "gap",  "std"),
            gap_ci_lo=("gap", _ci_lo),
            gap_ci_hi=("gap", _ci_hi),
            gap_p=(   "gap",  bootstrap_p_value),
            n_runs=(  "gap",  "size"),
        )
    )
    EPS = 1e-12
    pair_summary["gap_ci_width"] = pair_summary["gap_ci_hi"] - pair_summary["gap_ci_lo"]
    pair_summary["gap_snr"]      = pair_summary["gap_mean"] / (pair_summary["gap_std"].fillna(0.0) + EPS)
    pair_summary["gap_norm"]     = pair_summary["gap_mean"]
    return pair_summary.sort_values("gap_mean", ascending=False).reset_index(drop=True)


def matched_pair_bootstrap_summary(df_te, probs, pairs, *, thr=0.5, B=300):
    """
    CUB-style: for each (sid_A, sid_B, mpos, mneg) 4-tuple, runs matched_pair_eval B times.
    Returns (res_long, pair_summary).
    """
    _SCOLS = ["sid_A","sid_B","species_A","species_B","npos","nneg",
              "gap_mean","gap_std","gap_ci_lo","gap_ci_hi","gap_p",
              "gap_ci_width","gap_snr","gap_norm","n_runs"]
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=_SCOLS)
    rows = []
    for (a, b, mpos, mneg) in pairs:
        for boot_id in range(B):
            r = matched_pair_eval(df_te, probs,
                                  sid_A=int(a), sid_B=int(b),
                                  mpos=int(mpos), mneg=int(mneg),
                                  seed=int(boot_id), thr=float(thr))
            r["boot_id"] = int(boot_id)
            rows.append(r)
    res_long = pd.DataFrame(rows)
    if res_long.empty:
        return res_long, pd.DataFrame(columns=_SCOLS)
    return res_long, _build_pair_summary(res_long)


def fb_bootstrap_summary(df_te, probs, pairs, *, thr=0.5, B=300):
    """
    FunnyBirds variant: for each (sid_A, sid_B, mpos) 3-tuple, runs fb_pair_eval B times.
    Returns same column schema as matched_pair_bootstrap_summary so downstream is unchanged.
    """
    _SCOLS = ["sid_A","sid_B","species_A","species_B","npos","nneg",
              "gap_mean","gap_std","gap_ci_lo","gap_ci_hi","gap_p",
              "gap_ci_width","gap_snr","gap_norm","n_runs"]
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=_SCOLS)
    rows = []
    for (a, b, mpos) in pairs:
        for boot_id in range(B):
            r = fb_pair_eval(df_te, probs,
                             sid_A=int(a), sid_B=int(b),
                             mpos=int(mpos), seed=int(boot_id), thr=float(thr))
            r["boot_id"] = int(boot_id)
            rows.append(r)
    res_long = pd.DataFrame(rows)
    if res_long.empty:
        return res_long, pd.DataFrame(columns=_SCOLS)
    res_long = res_long.dropna(subset=["gap"])
    if res_long.empty:
        return res_long, pd.DataFrame(columns=_SCOLS)
    return res_long, _build_pair_summary(res_long)


# In[19]:


def species_recall_prevalence_table(df_te: pd.DataFrame, probs: np.ndarray, thr=0.5) -> pd.DataFrame:
    """
    Per-species table on the TEST set:
      n, n_pos, n_neg, prevalence, tp, recall, precision
    """
    df = df_te[["species_id", "species_name", "y"]].copy()
    df["prob"] = np.asarray(probs, dtype=float)
    df["pred"] = (df["prob"] >= thr).astype(int)
    g = (
        df.groupby(["species_id", "species_name"], as_index=False)
        .agg(n=("y","size"), n_pos=("y","sum"), n_pred_pos=("pred","sum"))
    )
    g["n_neg"]      = g["n"] - g["n_pos"]
    g["prevalence"] = g["n_pos"] / g["n"]
    tp = (
        df[df["y"] == 1]
        .groupby(["species_id", "species_name"])["pred"]
        .sum()
        .reset_index(name="tp")
    )
    out = g.merge(tp, on=["species_id", "species_name"], how="left")
    out["tp"]        = out["tp"].fillna(0).astype(int)
    out["recall"]    = np.where(out["n_pos"] > 0, out["tp"] / out["n_pos"], np.nan)
    out["precision"] = np.where(out["n_pred_pos"] > 0, out["tp"] / out["n_pred_pos"], np.nan)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def add_species_bootstrap_ci(
    df_te: pd.DataFrame, probs: np.ndarray,
    thr=0.5, B=300, min_pos_for_ci=1,
) -> pd.DataFrame:
    """
    Per-species bootstrap CI for recall.
    Bootstraps within each species by resampling that species' test images with replacement.
    Adds: recall_bs_mean, recall_ci_lo, recall_ci_hi, recall_ci_width.
    """
    df  = df_te[["species_id", "species_name", "y"]].copy()
    df["prob"] = np.asarray(probs, dtype=float)
    rows = []
    rng  = np.random.default_rng(0)
    for (sid, sname), d in df.groupby(["species_id", "species_name"]):
        d     = d.reset_index(drop=True)
        n     = len(d)
        n_pos = int(d["y"].sum())
        if n_pos < min_pos_for_ci:
            rows.append({"species_id": int(sid), "species_name": str(sname),
                         "recall_bs_mean": np.nan, "recall_ci_lo": np.nan,
                         "recall_ci_hi": np.nan, "recall_ci_width": np.nan, "B": int(B)})
            continue
        vals = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            s   = d.iloc[idx]
            pos = s[s["y"] == 1]
            if len(pos) == 0:
                vals.append(np.nan)
                continue
            vals.append(float((pos["prob"].to_numpy() >= thr).mean()))
        vals = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_ci(vals, alpha=0.05)
        rows.append({
            "species_id": int(sid), "species_name": str(sname),
            "recall_bs_mean": float(np.nanmean(vals)),
            "recall_ci_lo": lo, "recall_ci_hi": hi,
            "recall_ci_width": (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.nan,
            "B": int(B),
        })
    ci_df = pd.DataFrame(rows)
    base  = species_recall_prevalence_table(df_te, probs, thr=thr)
    return base.merge(ci_df, on=["species_id", "species_name"], how="left")


# In[20]:


def run_one_attribute(
    attr_name: str,
    feat_dir: Path,
    split_order_kind_train: str,
    split_order_train: np.ndarray,
    split_order_kind_test: str,
    split_order_test: np.ndarray,
    layer: str,
    *,
    min_certainty: int = 1,
    thr: float = 0.5,
    epochs: int = 25,
    min_each: int = 10,
    n_pairs: int = 200,
    B_gap: int = 100,
    B_species: int = 100,
    use_balanced_acc: bool = None,   # None = read EMERGE_METRIC at call time
):
    """
    Full pipeline for ONE attribute and ONE model's features.
    use_balanced_acc=None reads EMERGE_METRIC at call time so flipping
    the global switch propagates here automatically.
    """
    _use_bal = use_balanced_acc if use_balanced_acc is not None else (EMERGE_METRIC == "balanced")

    assert (split_order_kind_train == "image_id_like" and
            split_order_kind_test  == "image_id_like"), (
        "Cannot align features to attribute labels: labels_{split}.pt is not image_id-like."
    )
    # Build per-image labels
    aid       = attr_name_to_id[attr_name]
    lab       = build_attr_labeled_df(meta, img_attr_long, aid, min_certainty=min_certainty)
    lab_train = lab[lab["is_train"] == 1].copy()
    lab_test  = lab[lab["is_train"] == 0].copy()

    # Load precomputed features
    Xtr_all = load_features(feat_dir, layer, "train")
    Xte_all = load_features(feat_dir, layer, "test")

    # Align
    Xtr, df_tr = align_features_and_labels(Xtr_all, split_order_train, lab_train)
    Xte, df_te = align_features_and_labels(Xte_all, split_order_test,  lab_test)
    ytr = df_tr["y"].astype(int).to_numpy()
    yte = df_te["y"].astype(int).to_numpy()

    # Train probe + predict
    probe = train_probe(Xtr, ytr, seed=0, epochs=epochs)
    probs = predict_probs(probe, Xte)

    # Species table
    species_table = (
        add_species_bootstrap_ci(df_te, probs, thr=thr, B=B_species)
        if B_species > 0
        else species_recall_prevalence_table(df_te, probs, thr=thr)
    )

    # Overall accuracy
    if len(yte) == 0:
        test_acc = np.nan
    elif _use_bal:
        pred_bin = (probs >= thr).astype(int)
        tp = int(((pred_bin == 1) & (yte == 1)).sum())
        tn = int(((pred_bin == 0) & (yte == 0)).sum())
        fp = int(((pred_bin == 1) & (yte == 0)).sum())
        fn = int(((pred_bin == 0) & (yte == 1)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        test_acc = 0.5 * (tpr + tnr)
    else:
        test_acc = float(((probs >= thr).astype(int) == yte).mean())

    # Matched pairs — with automatic FunnyBirds fallback
    if n_pairs is None or int(n_pairs) <= 0:
        res_long     = pd.DataFrame()
        pair_summary = pd.DataFrame()
        pairs        = []
    else:
        pairs = make_candidate_pairs(df_te, min_each=min_each, max_pairs=n_pairs, seed=0)
        if len(pairs) > 0:
            res_long, pair_summary = matched_pair_bootstrap_summary(
                df_te, probs, pairs, thr=thr, B=B_gap
            )
        else:
            fb_pairs = make_candidate_pairs_fb(
                df_te, min_pos=max(1, min_each // 5), max_pairs=n_pairs
            )
            res_long, pair_summary = fb_bootstrap_summary(
                df_te, probs, fb_pairs, thr=thr, B=B_gap
            )
            pairs = fb_pairs

    mean_gap = float(pair_summary["gap_mean"].mean())        if len(pair_summary) else np.nan
    p90_gap  = float(pair_summary["gap_mean"].quantile(0.9)) if len(pair_summary) else np.nan

    info = {
        "attr":           attr_name,
        "layer":          layer,
        "n_train":        int(len(df_tr)),
        "n_test":         int(len(df_te)),
        "train_pos_rate": float(ytr.mean()) if len(ytr) else np.nan,
        "test_pos_rate":  float(yte.mean()) if len(yte) else np.nan,
        "test_acc":       float(test_acc),
        "test_acc_type":  "balanced" if _use_bal else "plain",
        "thr":            float(thr),
        "epochs":         int(epochs),
        "n_pairs":        int(len(pairs)),
        "B_gap":          int(B_gap),
        "B_species":      int(B_species),
        "mean_gap":       mean_gap,
        "p90_gap":        p90_gap,
    }
    return info, res_long, pair_summary, df_te, species_table


# In[21]:


def screen_attributes_for_species_variation(
    candidate_attrs,
    feat_dir: Path,
    kind_tr: str,  ids_tr: np.ndarray,
    kind_te: str,  ids_te: np.ndarray,
    layer: str,
    *,
    min_certainty: int = 1,
    thr: float = 0.5,
    min_pos_per_species: int  = 3,
    min_species_with_pos: int = 4,
    min_overall_prev: float   = 0.01,
    max_overall_prev: float   = 0.99,
    epochs: int = 8,
    max_attrs: int | None = None,
    verbose_every: int = 5,
    keep_error_examples: int = 5,
    B_species: int = 200,
    use_balanced_acc: bool = None,   # None = read EMERGE_METRIC at call time
):
    _use_bal = use_balanced_acc if use_balanced_acc is not None else (EMERGE_METRIC == "balanced")

    rows   = []
    errors = []
    stats  = {
        "tried": 0, "success": 0,
        "filtered_too_few_species_pos": 0,
        "filtered_prev_out_of_range":   0,
        "filtered_no_recall_vals":      0,
        "errored": 0,
    }
    cand = list(candidate_attrs)
    if max_attrs is not None:
        cand = cand[:max_attrs]

    for i, attr in enumerate(cand):
        stats["tried"] += 1
        try:
            info, _, _, _, species_table = run_one_attribute(
                attr, feat_dir, kind_tr, ids_tr, kind_te, ids_te,
                layer=layer, min_certainty=min_certainty, thr=thr,
                epochs=epochs, min_each=10, n_pairs=0, B_species=B_species,
                use_balanced_acc=_use_bal,
            )
            st            = species_table.copy()
            overall_prev  = float(st["n_pos"].sum() / st["n"].sum()) if st["n"].sum() > 0 else np.nan
            st_pos        = st[st["n_pos"] >= min_pos_per_species].copy()
            n_species_pos = int(len(st_pos))
            if n_species_pos < min_species_with_pos:
                stats["filtered_too_few_species_pos"] += 1
                continue
            if not (min_overall_prev <= overall_prev <= max_overall_prev):
                stats["filtered_prev_out_of_range"] += 1
                continue
            recall_vals = st_pos["recall"].dropna().to_numpy()
            if recall_vals.size == 0:
                stats["filtered_no_recall_vals"] += 1
                continue
            stats["success"] += 1
            recall_p90_p10 = float(np.quantile(recall_vals, 0.9) - np.quantile(recall_vals, 0.1))
            rows.append({
                "attr":            attr,
                "overall_prev":    overall_prev,
                "n_species_pos":   n_species_pos,
                "recall_std":      float(np.std(recall_vals)),
                "recall_range":    float(np.max(recall_vals) - np.min(recall_vals)),
                "recall_p90_p10":  recall_p90_p10,
                "test_acc":        float(info["test_acc"]),
                "test_acc_type":   info["test_acc_type"],
                "n_test":          int(info["n_test"]),
            })
            if verbose_every and ((i + 1) % verbose_every == 0):
                print(f"[{i+1}/{len(cand)}] ok: {attr}  "
                      f"prev={overall_prev:.3f}  n_species_pos={n_species_pos}  "
                      f"test_acc({info['test_acc_type']})={info['test_acc']:.3f}")
        except Exception as e:
            stats["errored"] += 1
            if len(errors) < keep_error_examples:
                errors.append((attr, repr(e)))
            continue

    screen_df = pd.DataFrame(rows)
    print("\n--- Screening summary ---")
    print(f"  acc_type used: {_use_bal and 'balanced' or 'plain'}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if errors:
        print("\nExample errors (first few):")
        for a, msg in errors:
            print("  ", a, "->", msg)
    if screen_df.empty:
        print("\nNo attributes passed filters.")
        return screen_df
    return screen_df.sort_values(
        ["recall_p90_p10", "recall_range", "recall_std"], ascending=False
    ).reset_index(drop=True)


# In[22]:


# ── Run screening ──────────────────────────────────────────────────────────────
_screen_feat = CBM_FEATS  if CBM_FEATS.exists()  else BASE_FEATS
_screen_ktr, _screen_itr = (cbm_kind_tr, cbm_ids_tr) if CBM_FEATS.exists() else (base_kind_tr, base_ids_tr)
_screen_kte, _screen_ite = (cbm_kind_te, cbm_ids_te) if CBM_FEATS.exists() else (base_kind_te, base_ids_te)

CANDIDATE_ATTRS = attr_df["attr_name"].tolist()

screen_df = screen_attributes_for_species_variation(
    CANDIDATE_ATTRS,
    feat_dir=_screen_feat,
    kind_tr=_screen_ktr, ids_tr=_screen_itr,
    kind_te=_screen_kte, ids_te=_screen_ite,
    layer=LAYER,
    min_certainty=1, thr=0.5,
    min_pos_per_species=3, min_species_with_pos=4,
    min_overall_prev=0.01, max_overall_prev=0.99,
    epochs=8, max_attrs=None, verbose_every=5,
)

# Fixed: use SCREENED_ATTR_LIST — do NOT overwrite ATTR_LIST from In[5]
if not screen_df.empty:
    SCREENED_ATTR_LIST = screen_df["attr"].tolist()
    print(f"\nFinal SCREENED_ATTR_LIST ({len(SCREENED_ATTR_LIST)} concepts):")
    for a in SCREENED_ATTR_LIST:
        print(" ", a)
    screen_df.head(26)
else:
    print("[warn] screening removed all attributes — falling back to full ATTR_LIST")
    SCREENED_ATTR_LIST = ATTR_LIST.copy()

attr_to_spread = screen_df.set_index("attr")["recall_p90_p10"].to_dict()


# In[23]:


def run_many(
    attr_list, model_name, feat_dir,
    kind_tr, ids_tr, kind_te, ids_te,
    *,
    layer,
    min_certainty=1, thr=0.5, epochs=25,
    min_each=3, n_pairs=200, B_gap=100, B_species=100,
    use_balanced_acc=None,   # None = read EMERGE_METRIC at call time
):
    """
    Runs run_one_attribute over a list of attrs.
    use_balanced_acc=None reads EMERGE_METRIC at call time.
    Returns: info_df, pairs_df, species_df
    """
    _use_bal = use_balanced_acc if use_balanced_acc is not None else (EMERGE_METRIC == "balanced")

    all_info, all_pair_summ, all_species = [], [], []
    for attr in attr_list:
        info, _, pair_summ, _, species_table = run_one_attribute(
            attr, feat_dir, kind_tr, ids_tr, kind_te, ids_te,
            layer=layer, min_certainty=min_certainty, thr=thr,
            epochs=epochs, min_each=min_each, n_pairs=n_pairs,
            B_gap=B_gap, B_species=B_species,
            use_balanced_acc=_use_bal,
        )
        info = dict(info)
        info["model"] = model_name
        all_info.append(info)
        if pair_summ is not None and len(pair_summ):
            ps = pair_summ.copy()
            ps["attr"] = attr; ps["model"] = model_name
            all_pair_summ.append(ps)
        st = species_table.copy()
        st["attr"] = attr; st["model"] = model_name
        all_species.append(st)
        print(model_name, attr,
              f"test_acc({info['test_acc_type']})=", round(info["test_acc"], 4),
              "mean_gap=", round(info["mean_gap"], 4))
    info_df    = pd.DataFrame(all_info)
    pairs_df   = pd.concat(all_pair_summ, ignore_index=True) if all_pair_summ else pd.DataFrame()
    species_df = pd.concat(all_species,   ignore_index=True) if all_species   else pd.DataFrame()
    return info_df, pairs_df, species_df


# In[ ]:


baseline_info, baseline_pairs, baseline_species = run_many(
    SCREENED_ATTR_LIST, "baseline", BASE_FEATS,
    base_kind_tr, base_ids_tr,
    base_kind_te, base_ids_te,
    layer=LAYER, thr=0.5, n_pairs=200, B_gap=100, B_species=100,
)

cbm_info, cbm_pairs, cbm_species = run_many(
    SCREENED_ATTR_LIST, "cbm", CBM_FEATS,
    cbm_kind_tr, cbm_ids_tr,
    cbm_kind_te, cbm_ids_te,
    layer=LAYER, thr=0.5, n_pairs=200, B_gap=100, B_species=100,
)

assert baseline_info["test_acc_type"].nunique() == 1, "Mixed acc types in baseline_info!"
assert cbm_info["test_acc_type"].nunique() == 1,      "Mixed acc types in cbm_info!"
print(f"test_acc type used: {baseline_info['test_acc_type'].iloc[0]}")

baseline_info, cbm_info


# In[ ]:


baseline_species.to_csv("baseline_species_fb.csv", index=False)
cbm_species.to_csv("cbm_species_fb.csv", index=False)
baseline_species.sort_values("tp", ascending=False).head(30)


# In[ ]:


cbm_species.sort_values("tp", ascending=False).head(30)


# In[ ]:


def summarize_by_attr(pairs_df: pd.DataFrame):
    """
    Collapses per-(attr, species pair) into per-attribute summary.
    Output: gap_mean, gap_median, gap_max, n_pairs,
            frac_p_small, frac_ci_above0, gap_snr_mean
    """
    if pairs_df.empty:
        return pairs_df
    tmp = pairs_df.copy()
    tmp["_ci_above0"] = tmp["gap_ci_lo"] > 0
    tmp["_p_small"]   = tmp["gap_p"] <= 0.05
    return (
        tmp.groupby(["model", "attr"], as_index=False)
        .agg(
            gap_mean=(       "gap_mean", "mean"),
            gap_median=(     "gap_mean", "median"),
            gap_max=(        "gap_mean", "max"),
            n_pairs=(        "gap_mean", "size"),
            frac_p_small=(   "_p_small",   "mean"),
            frac_ci_above0=( "_ci_above0", "mean"),
            gap_snr_mean=(   "gap_snr",    "mean"),
        )
        .sort_values(["model", "gap_mean"], ascending=[True, False])
    )


def top_pairs(pairs_df: pd.DataFrame, model: str, attr: str, k=10):
    """
    Returns top-k pairs by gap_mean for a given (model, attr).

    Column guide:
      gap_ci_lo/hi : bootstrap CI — if excludes 0, gap is stable
      gap_p        : bootstrap p-value for H0: gap==0 (two-sided)
      gap_snr      : gap_mean / gap_std — higher = more stable
      gap_norm     : gap_mean on 0..1 scale
    """
    sub  = pairs_df[(pairs_df["model"] == model) & (pairs_df["attr"] == attr)].copy()
    if sub.empty:
        return sub
    cols = ["species_A", "species_B",
            "gap_mean", "gap_ci_lo", "gap_ci_hi", "gap_p",
            "gap_std", "gap_snr", "gap_norm", "npos", "nneg", "n_runs"]
    cols = [c for c in cols if c in sub.columns]
    return sub.sort_values("gap_mean", ascending=False).head(k)[cols]


# In[ ]:


summary = pd.concat([
    summarize_by_attr(baseline_pairs),
    summarize_by_attr(cbm_pairs),
], ignore_index=True)
summary


# In[ ]:


for a in SCREENED_ATTR_LIST:
    print(f"\nAttribute: {a}")
    print("Baseline top pairs:")
    display(top_pairs(baseline_pairs, "baseline", a, k=10))
    print("CBM top pairs:")
    display(top_pairs(cbm_pairs, "cbm", a, k=10))


# In[ ]:


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def frac_of_final_idx(vals: np.ndarray, frac: float = 0.90) -> int:
    """Earliest index i such that vals[i] >= frac * vals[-1]."""
    v = pd.Series(np.asarray(vals, dtype=float)).ffill().bfill().to_numpy()
    target = frac * float(v[-1])
    for i, x in enumerate(v):
        if float(x) >= target:
            return int(i)
    return int(len(v) - 1)


def sharp_rise_idx(vals: np.ndarray) -> int:
    """Index of the largest single-step increase in vals."""
    v = pd.Series(np.asarray(vals, dtype=float)).ffill().bfill().to_numpy()
    diffs = np.diff(v)
    return int(np.argmax(diffs)) + 1


def train_linear_probe_multiclass(
    Xtr, ytr, Xte, yte, *, epochs=6, lr=3e-3, wd=1e-4, seed=0, device=None,
):
    """Multiclass linear probe — returns test accuracy."""
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long,    device=device)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    d, C  = Xtr_t.shape[1], int(ytr_t.max().item()) + 1
    model = nn.Linear(d, C).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(dim=1).detach().cpu().numpy()
    return float((pred == yte).mean())


def train_linear_probe_binary_weighted(
    Xtr, ytr, Xte, yte, *,
    epochs=8, lr=3e-3, wd=1e-4, seed=0, threshold=0.5, device=None,
):
    """
    Binary linear probe with pos_weight for class imbalance.
    Returns: (balanced_acc, plain_acc, pred_pos_rate_test, probs_test)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    ytr_np = np.asarray(ytr, dtype=np.int32).reshape(-1)
    yte_np = np.asarray(yte, dtype=np.int32).reshape(-1)
    Xtr_t  = torch.as_tensor(Xtr, dtype=torch.float32).to(device)
    Xte_t  = torch.as_tensor(Xte, dtype=torch.float32).to(device)
    ytr_t  = torch.as_tensor(ytr_np, dtype=torch.float32).view(-1, 1).to(device)
    d      = int(Xtr_t.shape[1])
    model  = nn.Linear(d, 1).to(device)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos    = float(ytr_np.sum())
    neg    = float(len(ytr_np) - ytr_np.sum())
    if pos <= 0:
        probs = np.zeros_like(yte_np, dtype=float)
        pred  = np.zeros_like(yte_np, dtype=int)
        return 0.5, float((pred == yte_np).mean()), 0.0, probs
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).view(-1).cpu().numpy()
    pred = (probs >= threshold).astype(int)
    ba   = balanced_accuracy(yte_np, pred)
    return float(ba), float((pred == yte_np).mean()), float(pred.mean()), probs


# In[ ]:


# Runs a multiclass linear probe (50 FunnyBirds species) at each layer
# to find the layer where species identity "emerges" in the backbone.

species_curve_ba = []
for layer in LAYERS:
    acc = train_linear_probe_multiclass(
        aligned_base_tr[layer], ysp_tr_ba,
        aligned_base_te[layer], ysp_te_ba,
        epochs=15, seed=0,
    )
    species_curve_ba.append(acc)

species_curve_ba      = np.asarray(species_curve_ba, dtype=float)
species_emerge_idx_ba = sharp_rise_idx(species_curve_ba)

print("Species emergence (baseline, sharp-jump):",
      species_emerge_idx_ba, "->", LAYERS[species_emerge_idx_ba])
for l, a in zip(LAYERS, species_curve_ba):
    print(f"  {l:12s}  {a:.4f}")

# ── Inline plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(species_curve_ba, marker="o", linewidth=2, label="Species ID accuracy")
ax.axvline(species_emerge_idx_ba, color="red", linestyle="--", linewidth=1.5,
           label=f"Sharp-jump: {LAYERS[species_emerge_idx_ba]}")
ax.set_xticks(range(len(LAYERS)))
ax.set_xticklabels(LAYERS, rotation=45, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Test accuracy (multiclass)")
ax.set_title("Species identity emergence across layers — Baseline ResNet (FunnyBirds)")
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:


def attribute_emergence(
    attr_list, *, min_certainty=1, epochs=8, seed=0, metric=None, frac=None,
):
    """
    Computes attribute emergence indices across LAYERS.
    Uses aligned_base_tr/te (pre-sliced) and aligned_tr/te_ids.

    metric : "balanced" | "plain" | None (reads EMERGE_METRIC)
    frac   : float | None (reads EMERGE_FRAC)
    """
    _metric = metric if metric is not None else EMERGE_METRIC
    _frac   = frac   if frac   is not None else EMERGE_FRAC

    attr_labels_tr, attr_labels_te = {}, {}
    for attr in attr_list:
        aid   = attr_name_to_id[attr]
        lab   = build_attr_labeled_df(meta, img_attr_long, aid, min_certainty=min_certainty)
        lab_i = lab.set_index("image_id")

        # Fixed: assert full coverage before .loc to catch silent NaN → int corruption
        missing_tr = set(aligned_tr_ids.tolist()) - set(lab_i.index.tolist())
        missing_te = set(aligned_te_ids.tolist()) - set(lab_i.index.tolist())
        assert not missing_tr, (
            f"{attr}: {len(missing_tr)} train image_ids missing from concept labels. "
            "Run prepare_funnybirds_metadata.py to regenerate metadata."
        )
        assert not missing_te, (
            f"{attr}: {len(missing_te)} test image_ids missing from concept labels. "
            "Run prepare_funnybirds_metadata.py to regenerate metadata."
        )

        attr_labels_tr[attr] = lab_i.loc[aligned_tr_ids, "y"].to_numpy(dtype=np.int32)
        attr_labels_te[attr] = lab_i.loc[aligned_te_ids, "y"].to_numpy(dtype=np.int32)

    rows = []
    for attr in attr_list:
        ytr, yte    = attr_labels_tr[attr], attr_labels_te[attr]
        ba_curve, acc_curve = [], []
        for layer in LAYERS:
            ba, acc, _, _ = train_linear_probe_binary_weighted(
                aligned_base_tr[layer], ytr,
                aligned_base_te[layer], yte,
                epochs=epochs, seed=seed,
            )
            ba_curve.append(ba); acc_curve.append(acc)
        ba_arr      = np.asarray(ba_curve,  dtype=float)
        acc_arr     = np.asarray(acc_curve, dtype=float)
        _emerge_arr = ba_arr if _metric == "balanced" else acc_arr
        e_jump = sharp_rise_idx(_emerge_arr)
        e_frac = frac_of_final_idx(_emerge_arr, frac=_frac)
        rows.append({
            "attr":              attr,
            "emerge_idx_jump":   int(e_jump),
            "emerge_layer_jump": LAYERS[e_jump],
            "emerge_idx_frac":   int(e_frac),
            "emerge_layer_frac": LAYERS[e_frac],
            "final_ba":          float(ba_arr[-1]),
            "final_acc":         float(acc_arr[-1]),
            "emerge_metric":     _metric,
            "emerge_frac":       _frac,
        })
        print(f"  {attr}: frac{int(_frac*100)}={LAYERS[e_frac]}  "
              f"jump={LAYERS[e_jump]}  final_acc={acc_arr[-1]:.3f}")
    return pd.DataFrame(rows)


attr_emerge_df = attribute_emergence(SCREENED_ATTR_LIST)  # fixed: was ATTR_LIST
print()
print(attr_emerge_df[["attr", "emerge_layer_frac", "emerge_layer_jump", "final_ba", "final_acc"]])


# In[ ]:


def plot_attribute_curve(attr: str, title_extra="", metric=None, frac=None):
    """
    Plots balanced accuracy, plain accuracy, and predicted positive rate across LAYERS
    for a single FunnyBirds concept, using a binary weighted linear probe.

    metric : "balanced" | "plain" | None (reads EMERGE_METRIC)
    frac   : float | None (reads EMERGE_FRAC)
    Prints inline.
    """
    _metric = metric if metric is not None else EMERGE_METRIC
    _frac   = frac   if frac   is not None else EMERGE_FRAC

    aid   = attr_name_to_id[attr]
    lab   = build_attr_labeled_df(meta, img_attr_long, aid, min_certainty=1)
    lab_i = lab.set_index("image_id")
    ytr   = lab_i.loc[aligned_tr_ids, "y"].to_numpy(dtype=np.int32)
    yte   = lab_i.loc[aligned_te_ids, "y"].to_numpy(dtype=np.int32)

    ba_curve, acc_curve, ppos_curve = [], [], []
    for layer in LAYERS:
        ba, acc, ppos, _ = train_linear_probe_binary_weighted(
            aligned_base_tr[layer], ytr,
            aligned_base_te[layer], yte,
            epochs=8, seed=0,
        )
        ba_curve.append(ba); acc_curve.append(acc); ppos_curve.append(ppos)

    ba_arr      = np.asarray(ba_curve,  float)
    acc_arr     = np.asarray(acc_curve, float)
    _emerge_arr = ba_arr if _metric == "balanced" else acc_arr
    e_jump = sharp_rise_idx(_emerge_arr)
    e_frac = frac_of_final_idx(_emerge_arr, frac=_frac)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ba_arr,          marker="o", label="Balanced accuracy (BA)")
    ax.plot(acc_arr,         marker="o", label="Plain accuracy")
    ax.plot(ppos_curve,      marker="o", label="Predicted positive rate (test)")
    ax.axvline(e_jump, color="red",    linestyle="--", linewidth=1.2,
               label=f"sharp_jump: {LAYERS[e_jump]}")
    ax.axvline(e_frac, color="orange", linestyle=":",  linewidth=1.2,
               label=f"frac{int(_frac*100)}: {LAYERS[e_frac]}")
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(LAYERS, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Concept: {attr}  |  metric={_metric}  frac={_frac}\n"
        f"jump={LAYERS[e_jump]},  frac{int(_frac*100)}={LAYERS[e_frac]}  {title_extra}"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


# Examples — first and last concept in ATTR_LIST
plot_attribute_curve(SCREENED_ATTR_LIST[1])
plot_attribute_curve(SCREENED_ATTR_LIST[-1])


# In[ ]:


SCREENED_ATTR_LIST


# In[ ]:


plot_attribute_curve(SCREENED_ATTR_LIST[-5])


# In[ ]:


plot_attribute_curve(SCREENED_ATTR_LIST[-4])


# In[ ]:


print(f"EMERGE_METRIC    : {EMERGE_METRIC}")
print(f"EMERGE_FRAC      : {EMERGE_FRAC}")
print(f"EMERGE_CRITERION : {EMERGE_CRITERION}")
print(f"Species emergence layer (baseline): {LAYERS[species_emerge_idx_ba]}")
print()

for crit, col in [("frac90", "emerge_idx_frac"), ("sharp_jump", "emerge_idx_jump")]:
    at_   = attr_emerge_df[attr_emerge_df[col] == species_emerge_idx_ba]
    after = attr_emerge_df[attr_emerge_df[col] >  species_emerge_idx_ba]
    pre   = attr_emerge_df[attr_emerge_df[col] <  species_emerge_idx_ba]
    print(f"[{crit}]  PRE={len(pre)}  AT={len(at_)}  AFTER={len(after)}")

attr_emerge_df["group"] = "at"
attr_emerge_df.loc[attr_emerge_df[emerge_col] < species_emerge_idx_ba, "group"] = "pre"
attr_emerge_df.loc[attr_emerge_df[emerge_col] > species_emerge_idx_ba, "group"] = "post"

print(f"\nUsing criterion : {EMERGE_CRITERION}")
print("  pre: ",  (attr_emerge_df["group"] == "pre").sum(),
      "  at: ",   (attr_emerge_df["group"] == "at").sum(),
      "  post: ", (attr_emerge_df["group"] == "post").sum())
print()
print(attr_emerge_df[["attr", layer_col, "group", "final_ba", "final_acc",
                       "emerge_metric", "emerge_frac"]])


# In[ ]:


import os
os.makedirs("../figures", exist_ok=True)

for model_name, info_df in [("baseline", baseline_info), ("cbm", cbm_info)]:
    if info_df.empty or "attr" not in info_df.columns:
        print(f"[{model_name}] info_df empty — run run_many first")
        continue
    grp = info_df.merge(
        attr_emerge_df[["attr", "group", emerge_col, layer_col]],
        on="attr", how="left",
    )
    print(f"\n[{model_name}] Attributes by emergence group:")
    print(grp["group"].value_counts())
    print(f"\n[{model_name}] Mean gap by emergence group:")
    print(grp.groupby("group")["mean_gap"].agg(["mean", "std", "count"]))

    order = [g for g in ["pre", "at", "post"] if g in grp["group"].values]

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=grp, x="group", y="mean_gap",
                       order=order, inner="box", cut=0, ax=ax)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Emergence group (criterion: {EMERGE_CRITERION})")
        ax.set_ylabel("Mean matched-pair recall gap")
        ax.set_title(f"Recall gap by attribute emergence group\n"
                     f"({model_name} — FunnyBirds)")
        plt.tight_layout()
        plt.savefig(f"../figures/recall_gap_by_group_fb_{model_name}.pdf", bbox_inches="tight")
        plt.show()
    except Exception as e:
        print(f"seaborn not available ({e}), using scatter fallback")
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in order:
            vals = grp[grp["group"] == g]["mean_gap"].dropna().values
            ax.scatter([g] * len(vals), vals, alpha=0.6, label=g)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set_ylabel("Mean matched-pair recall gap")
        ax.set_title(f"Recall gap by group ({model_name} — FunnyBirds)")
        plt.tight_layout()
        plt.savefig(f"../figures/recall_gap_by_group_fb_{model_name}.pdf", bbox_inches="tight")
        plt.show()


# In[ ]:


# ── Better visualization: ECDF + proportion discriminative + scatter trend ─────
# Three views per model:
#   1. ECDF of gap_mean by group       — heavier right tail in POST = signal
#   2. % attributes discriminative     — proportion test, what we actually care about
#   3. Scatter: emergence index vs gap — continuous, no binning, trend line

_egrp       = attr_emerge_df[["attr", "group", emerge_col]].drop_duplicates("attr")
summary_grp = summary.merge(_egrp, on="attr", how="left")
group_order  = [g for g in ["pre", "at", "post"] if g in summary_grp["group"].dropna().values]
group_colors = {"pre": "#5BA4CF", "at": "#F5A623", "post": "#D0021B"}

for model_name in ["baseline", "cbm"]:
    sg = summary_grp[summary_grp["model"] == model_name].copy()
    if sg.empty:
        print(f"[{model_name}] no summary rows — skipping")
        continue

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Plot 1: ECDF ────────────────────────────────────────────────────────────
    ax = axes[0]
    for g in group_order:
        vals = sg[sg["group"] == g]["gap_mean"].dropna().values
        if len(vals) == 0:
            continue
        vals_s = np.sort(vals)
        ecdf   = np.arange(1, len(vals_s) + 1) / len(vals_s)
        ax.step(vals_s, ecdf, where="post",
                label=f"{g} (n={len(vals)})",
                color=group_colors.get(g), linewidth=2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean matched-pair recall gap")
    ax.set_ylabel("Cumulative fraction of attributes")
    ax.set_title("ECDF of recall gaps by group\n(POST right-shifted = entanglement signal)")
    ax.legend()

    # ── Plot 2: % discriminative ────────────────────────────────────────────────
    ax     = axes[1]
    ci_col = "frac_ci_above0"
    if ci_col in sg.columns:
        disc_rows = []
        for g in group_order:
            sub = sg[sg["group"] == g][ci_col].dropna()
            if len(sub) == 0:
                continue
            disc_rows.append({
                "group": g,
                "frac_discriminative": float((sub > 0.5).mean()),
                "n": len(sub),
            })
        disc_df = pd.DataFrame(disc_rows)
        bars = ax.bar(
            disc_df["group"], disc_df["frac_discriminative"],
            color=[group_colors.get(g, "grey") for g in disc_df["group"]],
            edgecolor="black", linewidth=0.8,
        )
        for bar, row in zip(bars, disc_df.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"n={row.n}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="50% line")
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Emergence group (criterion: {EMERGE_CRITERION})")
        ax.set_ylabel("Fraction of attributes\n(>50% pairs: CI entirely above 0)")
        ax.set_title("Proportion discriminative by group\n(what we actually care about)")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "frac_ci_above0 not available\n(run summarize_by_attr first)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Proportion discriminative")

    # ── Plot 3: Scatter — emergence index vs gap_mean ───────────────────────────
    ax = axes[2]
    if emerge_col in sg.columns:
        for g in group_order:
            sub = sg[sg["group"] == g].dropna(subset=["gap_mean", emerge_col])
            if len(sub) == 0:
                continue
            ax.scatter(sub[emerge_col], sub["gap_mean"],
                       label=f"{g} (n={len(sub)})",
                       color=group_colors.get(g), alpha=0.65, s=40, edgecolors="none")
        all_v = sg.dropna(subset=["gap_mean", emerge_col])
        if len(all_v) > 5:
            xs = all_v[emerge_col].values.astype(float)
            ys = all_v["gap_mean"].values
            m, b = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xline, m * xline + b, color="black", linestyle="--",
                    linewidth=1.5, label=f"trend (slope={m:.4f})")
        ax.axvline(species_emerge_idx_ba, color="black", linestyle=":", linewidth=1.2,
                   label=f"species layer ({LAYERS[species_emerge_idx_ba]})")
        ax.set_xlabel(f"Attribute emergence layer index ({EMERGE_CRITERION})")
        ax.set_ylabel("Mean matched-pair recall gap")
        ax.set_title("Later-emerging → larger gaps?\n(continuous, no binning)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"Column '{emerge_col}' not in data",
                ha="center", va="center", transform=ax.transAxes)

    plt.suptitle(
        f"Entanglement signal: recall gap vs. attribute emergence timing\n"
        f"({model_name} — FunnyBirds — criterion: {EMERGE_CRITERION}, n_attrs={len(sg)})",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"../figures/recall_gap_entanglement_fb_{model_name}.pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved: ../figures/recall_gap_entanglement_fb_{model_name}.pdf")


# In[ ]:


# ── Benjamini-Hochberg FDR correction on per-pair p-values ────────────────────
try:
    from statsmodels.stats.multitest import multipletests
    for name, df in [("baseline", baseline_pairs), ("cbm", cbm_pairs)]:
        if df.empty or "gap_p" not in df.columns:
            print(f"[{name}] No pairs data available for FDR correction")
            continue
        pvals = df["gap_p"].fillna(1.0).to_numpy()
        reject, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
        df["gap_p_fdr"]   = pvals_fdr
        df["gap_sig_fdr"] = reject
        n_sig = int(reject.sum())
        print(f"[{name}] FDR-corrected pairs (p < 0.05): {n_sig} / {len(df)}")
    print("Use gap_p_fdr < 0.05 as the significance threshold for reporting.")
except ImportError:
    print("statsmodels not available; install with: pip install statsmodels")
    print("Raw p-values in gap_p column (uncorrected for multiple testing)")


# In[ ]:


grp["group"].values


# In[ ]:


def species_attr_gap(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form: one row per (species, attr). Shows which attributes drive each species' gap."""
    if pairs_df.empty:
        return pairs_df
    a_side = pairs_df[["attr", "sid_A", "species_A", "gap_mean"]].rename(
        columns={"sid_A": "species_id", "species_A": "species_name"}
    )
    b_side = pairs_df[["attr", "sid_B", "species_B", "gap_mean"]].rename(
        columns={"sid_B": "species_id", "species_B": "species_name"}
    )
    combined = pd.concat([a_side, b_side], ignore_index=True)
    return (
        combined.groupby(["species_id", "species_name", "attr"], as_index=False)
        .agg(mean_gap=("gap_mean", "mean"), n_pairs=("gap_mean", "size"))
        .sort_values(["species_name", "mean_gap"], ascending=[True, False])
        .reset_index(drop=True)
    )


# In[ ]:


import seaborn as sns

for name, ps in [("baseline", baseline_pairs), ("cbm", cbm_pairs)]:
    if ps.empty:
        continue
    sa    = species_attr_gap(ps)
    pivot = sa.pivot(index="species_name", columns="attr", values="mean_gap")

    fig, ax = plt.subplots(figsize=(len(pivot.columns) * 0.8 + 2, len(pivot) * 0.4 + 2))
    sns.heatmap(pivot, ax=ax, cmap="Reds", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "mean gap"})
    ax.set_title(f"{name} — mean recall gap per (species, attribute)")
    plt.tight_layout()
    plt.show()


# In[ ]:


def species_attr_recall(species_df: pd.DataFrame, min_pos: int = 3) -> pd.DataFrame:
    """
    For each (species, attr): recall, CI, and deviation from the median recall
    across all species that have the concept.
    Positive deviation = probe fires MORE on this species than average.
    Negative deviation = probe fires LESS.
    """
    df = species_df[species_df["n_pos"] >= min_pos].copy()

    # median recall per attr across all species that have the concept
    attr_median = (
        df.groupby("attr")["recall"]
        .median()
        .rename("attr_median_recall")
    )
    df = df.join(attr_median, on="attr")
    df["recall_dev"] = df["recall"] - df["attr_median_recall"]
    return df[["attr", "species_name", "n_pos", "recall",
               "recall_ci_lo", "recall_ci_hi", "recall_dev",
               "attr_median_recall"]]

def pairs_for_species(pairs_df: pd.DataFrame, species_name: str) -> pd.DataFrame:
    """
    All pairs involving species_name, showing the partner and gap per attr.
    """
    mask_a = pairs_df["species_A"] == species_name
    mask_b = pairs_df["species_B"] == species_name

    a_side = pairs_df[mask_a].copy()
    a_side["partner"] = a_side["species_B"]

    b_side = pairs_df[mask_b].copy()
    b_side["partner"] = b_side["species_A"]

    combined = pd.concat([a_side, b_side], ignore_index=True)
    cols = ["attr", "partner", "gap_mean", "gap_ci_lo", "gap_ci_hi", "gap_p", "npos", "n_runs"]
    cols = [c for c in cols if c in combined.columns]
    return (combined[cols]
            .sort_values("gap_mean", ascending=False)
            .reset_index(drop=True))


# In[ ]:


for name, sp_df in [("baseline", baseline_species), ("cbm", cbm_species)]:
    sar = species_attr_recall(sp_df)

    # rank species by recall variance across attrs — high variance = inconsistent probe
    sp_var = (
        sar.groupby("species_name")["recall_dev"]
        .agg(recall_dev_std="std", recall_dev_range=lambda x: x.max() - x.min(),
             n_attrs="size")
        .sort_values("recall_dev_std", ascending=False)
        .reset_index()
    )

    print(f"\n{'='*60}\n{name} — species ranked by recall inconsistency across attrs")
    display(sp_var.head(10))

    print(f"\n{name} — worst species: attrs driving deviation")
    worst = sp_var["species_name"].iloc[0]
    sub   = sar[sar["species_name"] == worst].sort_values("recall_dev")
    display(sub)


# In[ ]:


for name, ps in [("baseline", baseline_pairs), ("cbm", cbm_pairs)]:
    if ps.empty:
        continue
    sp_rank = per_species_gap(ps)
    print(f"\n{'='*60}\n{name}")
    for _, row in sp_rank.head(5).iterrows():
        sname = row["species_name"]
        print(f"\n  {sname}  (mean_gap={row['mean_gap']:.3f} across {int(row['n_pairs'])} pairs)")
        display(pairs_for_species(ps, sname).head(10))


# In[ ]:




