#!/usr/bin/env python3
"""Generate funnybird_notebooks/fb_cbm_counterfactual.ipynb.

Data-loading patterns taken verbatim from fb_recallv2.py / fb_recallv2.ipynb.
"""
import json
from pathlib import Path


def code(src: str):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def md(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

# ── Cell 0: title ─────────────────────────────────────────────────────────────
cells.append(md("""\
# FunnyBirds CBM — Counterfactual Concept Swap

**Motivation:** recall gaps in `fb_recallv2.ipynb` show that a concept probe fires
differently across species even when both truly have the concept.  But a recall gap is a
behavioural proxy — it could reflect visual confounding in the backbone rather than
species identity leaking *through the concept bottleneck itself*.

**This notebook tests the mechanism directly** using the CBM's own bottleneck and label head:

1. **Counterfactual swap** — For an all-positive pair (species A, B, concept C):
   take species-A test images → compute concept activations z_A ∈ ℝ^26 → replace
   z_A[C] with z_B[C] (same concept, different species donor) → run label head.
   If P(species B) rises: z_C was carrying species-B identity.

2. **Species-identity probe on z** — train a linear classifier `z → species_id`.
   If z alone predicts species, the bottleneck is leaking species identity.

3. **Per-concept binary discriminability** — for each concept C, train a binary
   classifier `z_C (scalar) → {A, B}`.  Above-chance = z_C distinguishes species
   beyond merely whether the concept is present.

Data-loading follows `fb_recallv2.ipynb` exactly."""))

# ── Cell 1: imports ───────────────────────────────────────────────────────────
cells.append(code("""\
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats"""))

# ── Cell 2: paths — exact same structure as fb_recallv2.py ───────────────────
cells.append(md("## 0. Paths and configuration"))

cells.append(code("""\
import sys
ROOT = Path('/scratch/network/cr7998/cv_emergence_project')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FB        = ROOT / 'data'     / 'FunnyBirds'
CBM_FEATS = ROOT / 'features' / 'resnet50_cbm_funnybirds'
CBM_CKPT  = ROOT / 'checkpoints_funnybirds' / 'cbm_funnybirds.pth'

assert FB.exists(),        f'Missing FunnyBirds folder: {FB}'
assert CBM_FEATS.exists(), f'Missing CBM features: {CBM_FEATS}'
assert CBM_CKPT.exists(),  f'Missing CBM checkpoint: {CBM_CKPT}'

N_SPECIES    = 50
N_CONCEPTS   = 26
PROBE_EPOCHS = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[config] device: {device}')
print(f'[config] CBM checkpoint: {CBM_CKPT}')"""))

# ── Cell 3: data-loading helpers — verbatim from fb_recallv2.py cell 3-7 ────
cells.append(md("## 1. Data-loading helpers  *(from fb_recallv2.py)*"))

cells.append(code("""\
def load_species_maps(fb_root: Path):
    \"\"\"Load species ID→name mappings from FunnyBirds metadata/classes.csv.\"\"\"
    classes_csv = fb_root / 'metadata' / 'classes.csv'
    if not classes_csv.exists():
        raise FileNotFoundError(
            'metadata/classes.csv not found. Run prepare_funnybirds_metadata.py first.')
    df = pd.read_csv(classes_csv)
    id2name  = dict(zip(df['class_id'], df['class_name']))
    id2short = {k: v.replace('funnybird_', 'FB') for k, v in id2name.items()}
    return id2name, id2short


def load_meta(fb_root: Path) -> pd.DataFrame:
    \"\"\"Per-image metadata: image_id, class_id, species_id, species_name, is_train.\"\"\"
    images_csv = fb_root / 'metadata' / 'images.csv'
    if not images_csv.exists():
        raise FileNotFoundError(
            'metadata/images.csv not found. Run prepare_funnybirds_metadata.py first.')
    df = pd.read_csv(images_csv)
    id2name, _ = load_species_maps(fb_root)
    df['species_id']   = df['class_id']
    df['species_name'] = df['class_id'].map(id2name)
    return df


def load_image_attr_labels_robust(fb_root: Path) -> pd.DataFrame:
    \"\"\"
    Long-form per-image concept labels: image_id, attr_id, attr_name, is_present, certainty.
    certainty is always 1 for FunnyBirds (ground-truth, no annotation noise).
    \"\"\"
    concepts_csv = fb_root / 'metadata' / 'image_concepts_binary.csv'
    if not concepts_csv.exists():
        raise FileNotFoundError(
            'metadata/image_concepts_binary.csv not found. '
            'Run prepare_funnybirds_metadata.py first.')
    wide = pd.read_csv(concepts_csv)
    concept_cols = [c for c in wide.columns if c != 'image_id']
    long = wide.melt(id_vars='image_id', value_vars=concept_cols,
                     var_name='attr_name', value_name='is_present')
    long['attr_id']    = long.groupby('attr_name', sort=False).ngroup()
    long['is_present'] = long['is_present'].astype(int)
    long['certainty']  = 1
    return long[['image_id', 'attr_id', 'attr_name', 'is_present', 'certainty']]


def load_attr_maps(fb_root: Path):
    \"\"\"Concept name↔id from metadata/concepts.csv.\"\"\"
    concepts_csv = fb_root / 'metadata' / 'concepts.csv'
    if not concepts_csv.exists():
        raise FileNotFoundError(
            'metadata/concepts.csv not found. Run prepare_funnybirds_metadata.py first.')
    df = pd.read_csv(concepts_csv)
    id2name = dict(zip(df['concept_id'], df['concept_name']))
    name2id = dict(zip(df['concept_name'], df['concept_id']))
    return id2name, name2id


def build_attr_labeled_df(
    meta: pd.DataFrame,
    img_attr_long: pd.DataFrame,
    attr_id: int,
    min_certainty: int = 1,
) -> pd.DataFrame:
    \"\"\"image_id, species_id, species_name, is_train, y, certainty for one concept.\"\"\"
    sub = img_attr_long[img_attr_long['attr_id'] == int(attr_id)].copy()
    sub = sub[sub['certainty'] >= int(min_certainty)].copy()
    out = meta.merge(sub[['image_id', 'is_present', 'certainty']], on='image_id', how='inner')
    out = out.rename(columns={'is_present': 'y'})
    out['y'] = out['y'].astype(int)
    return out[['image_id', 'species_id', 'species_name', 'is_train', 'y', 'certainty']]


print('Defined: load_species_maps  load_meta  load_image_attr_labels_robust')
print('         load_attr_maps  build_attr_labeled_df')"""))

# ── Cell 4: feature-loading helpers — verbatim from fb_recallv2.py cell 8 ───
cells.append(code("""\
def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_features(feat_dir: Path, layer: str, split: str) -> torch.Tensor:
    p = feat_dir / f'{layer}_{split}.pt'
    assert p.exists(), f'Missing: {p}'
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
        return 'species_id_like'
    if arr.max() > 200:
        return 'image_id_like'
    return 'unknown'


def load_split_order(feat_dir: Path, split: str):
    p = feat_dir / f'labels_{split}.pt'
    assert p.exists(), f'Missing: {p}'
    t = safe_torch_load(p)
    assert isinstance(t, dict), f'Expected dict in {p}, got {type(t)}'
    assert 'image_ids' in t, f'{p} missing image_ids; has {list(t.keys())}'
    ids  = to_1d_int_array(t['image_ids'])
    kind = infer_kind(ids)
    return kind, ids


def align_features_and_labels(
    X_split: torch.Tensor,
    image_ids_in_feature_order: np.ndarray,
    labeled_df_split: pd.DataFrame,
):
    \"\"\"
    Align feature rows to attribute labels by image_id.
    Returns X_aligned [M, D] and df_aligned [M rows] in the same order.
    \"\"\"
    labeled  = labeled_df_split.set_index('image_id')[['y', 'species_id', 'species_name']]
    keep_idx = []
    rows     = []
    for i, img_id in enumerate(image_ids_in_feature_order):
        img_id = int(img_id)
        if img_id in labeled.index:
            keep_idx.append(i)
            y, sid, sname = labeled.loc[img_id]
            rows.append((img_id, int(sid), str(sname), int(y)))
    X_aligned  = X_split[keep_idx]
    df_aligned = pd.DataFrame(rows, columns=['image_id', 'species_id', 'species_name', 'y'])
    return X_aligned, df_aligned


print('Defined: safe_torch_load  load_features  load_split_order')
print('         to_1d_int_array  infer_kind  align_features_and_labels')"""))

# ── Cell 5: concept names, class-concept matrix ───────────────────────────────
cells.append(md("## 2. Concept names, class-concept matrix, metadata"))

cells.append(code("""\
from datasets.funnybirds_dataset import FunnyBirdsDataset
from datasets.funnybirds_dataset import concept_names as _fb_concept_names

CONCEPT_NAMES  = _fb_concept_names()   # 26 strings: ['beak_0', 'beak_1', ...]
CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPT_NAMES)}

PART_GROUPS = {
    part: [i for i, c in enumerate(CONCEPT_NAMES) if c.startswith(f'{part}_')]
    for part in ['beak', 'wing', 'tail', 'foot', 'eye']
}
PART_COLORS = {
    'beak': 'steelblue', 'wing': 'seagreen', 'tail': 'crimson',
    'foot': 'darkorange', 'eye': 'purple',
}
CONCEPT_TO_PART = {
    c: part
    for part, idxs in PART_GROUPS.items()
    for c in [CONCEPT_NAMES[i] for i in idxs]
}

# Exact class-concept matrix — no annotation noise (key FunnyBirds advantage)
_fb_ds    = FunnyBirdsDataset(FB, split='train')
cc_matrix, _ = _fb_ds.get_class_concept_matrix()   # [50, 26] int tensor
cc_matrix = cc_matrix.numpy()

cc_df = pd.DataFrame(
    cc_matrix,
    columns=CONCEPT_NAMES,
    index=[f'funnybird_{i:02d}' for i in range(N_SPECIES)],
)
print(f'Concepts ({len(CONCEPT_NAMES)}): {CONCEPT_NAMES}')
print(f'CC matrix: {cc_matrix.shape}  — each row sums to {cc_matrix.sum(1).mean():.0f}')
cc_df.head()"""))

# ── Cell 6: load metadata ────────────────────────────────────────────────────
cells.append(code("""\
# Load metadata and attribute maps — same as fb_recallv2.py cell 6
meta          = load_meta(FB)
img_attr_long = load_image_attr_labels_robust(FB)
attr_id_to_name, attr_name_to_id = load_attr_maps(FB)

attr_df = pd.DataFrame({
    'attr_name': list(attr_name_to_id.keys()),
    'attr_id':   list(attr_name_to_id.values()),
})

id2name, _ = load_species_maps(FB)
def spname(sid: int) -> str:
    return id2name.get(int(sid), f'funnybird_{int(sid):02d}')

print(f'meta:          {len(meta)} images  '
      f'({meta["is_train"].sum()} train, {(meta["is_train"]==0).sum()} test)')
print(f'img_attr_long: {len(img_attr_long)} rows, '
      f'{img_attr_long["attr_name"].nunique()} concepts')
print(f'Concepts: {list(attr_name_to_id.keys())}')"""))

# ── Cell 7: load split order ──────────────────────────────────────────────────
cells.append(code("""\
# Load split ordering (image_ids in DataLoader order)
cbm_kind_tr, cbm_ids_tr = load_split_order(CBM_FEATS, 'train')
cbm_kind_te, cbm_ids_te = load_split_order(CBM_FEATS, 'test')

print(f'CBM train: {cbm_kind_tr}  id range: ({cbm_ids_tr.min()}, {cbm_ids_tr.max()})')
print(f'CBM test:  {cbm_kind_te}  id range: ({cbm_ids_te.min()}, {cbm_ids_te.max()})')
print(f'Train N={len(cbm_ids_tr)}  Test N={len(cbm_ids_te)}')"""))

# ── Cell 8: load CBM checkpoint ───────────────────────────────────────────────
cells.append(md("""\
## 3. Load CBM weights and compute concept activations

Architecture: `backbone → avgpool [2048] → concept_head [26] → sigmoid → label_head [50]`

- **avgpool features** already extracted at `CBM_FEATS/avgpool_{split}.pt`
- **concept_head** (`W_c [26, 2048]`, `b_c [26]`) and **label_head** (`W_y [50, 26]`,
  `b_y [50]`) from checkpoint

`z = sigmoid(avgpool @ W_c.T + b_c)` — the 26-dim concept bottleneck activation."""))

cells.append(code("""\
ckpt = safe_torch_load(CBM_CKPT)
sd   = ckpt.get('model_state_dict', ckpt)

W_c = sd['concept_head.weight'].float()   # [26, 2048]
b_c = sd['concept_head.bias'].float()     # [26]
W_y = sd['label_head.weight'].float()     # [50, 26]
b_y = sd['label_head.bias'].float()       # [50]

print(f'concept_head: {tuple(W_c.shape)}  bias: {tuple(b_c.shape)}')
print(f'label_head:   {tuple(W_y.shape)}  bias: {tuple(b_y.shape)}')
cfg = ckpt.get('config', {})
if cfg:
    print(f'config: {cfg}')"""))

# ── Cell 9: load avgpool, compute z ──────────────────────────────────────────
cells.append(code("""\
# Load avgpool features and compute z for train + test splits
avg_te = load_features(CBM_FEATS, 'avgpool', 'test')
avg_tr = load_features(CBM_FEATS, 'avgpool', 'train')

# Align metadata to feature order (same pattern as fb_recallv2.py)
meta_tr = meta[meta['is_train'] == 1].copy()
meta_te = meta[meta['is_train'] == 0].copy()

# Use any concept to get the aligned species_ids — we just need the image_id → species_id map
_meta_idx = meta.set_index('image_id')
sids_te = np.array([int(_meta_idx.loc[int(i), 'species_id']) for i in cbm_ids_te])
sids_tr = np.array([int(_meta_idx.loc[int(i), 'species_id']) for i in cbm_ids_tr])

@torch.no_grad()
def compute_z(avgpool: torch.Tensor) -> torch.Tensor:
    \"\"\"z = sigmoid(avgpool @ W_c.T + b_c)  [N, 26]\"\"\"
    return torch.sigmoid(avgpool @ W_c.T + b_c)

@torch.no_grad()
def compute_logits(z: torch.Tensor) -> torch.Tensor:
    \"\"\"logits = z @ W_y.T + b_y  [N, 50]\"\"\"
    return z @ W_y.T + b_y

z_te = compute_z(avg_te)   # [N_te, 26]
z_tr = compute_z(avg_tr)   # [N_tr, 26]

print(f'avgpool_te: {tuple(avg_te.shape)}  →  z_te: {tuple(z_te.shape)}')
print(f'avgpool_tr: {tuple(avg_tr.shape)}  →  z_tr: {tuple(z_tr.shape)}')"""))

# ── Cell 10: sanity checks ────────────────────────────────────────────────────
cells.append(code("""\
# Sanity: species accuracy
logits_te = compute_logits(z_te)
pred_sp   = logits_te.argmax(dim=1).numpy()
sp_acc    = float((pred_sp == sids_te).mean())
print(f'CBM test species accuracy: {sp_acc:.4f}')

# Sanity: concept accuracy vs GT class-concept matrix
gt_concepts  = cc_matrix[sids_te]                      # [N_te, 26]
pred_concepts = (z_te.numpy() > 0.5).astype(int)
print(f'CBM concept accuracy (threshold=0.5): {(pred_concepts == gt_concepts).mean():.4f}')
print('Per-concept:')
for name, acc in zip(CONCEPT_NAMES, (pred_concepts == gt_concepts).mean(axis=0)):
    print(f'  {name:12s}: {acc:.3f}')"""))

# ── Cell 11: counterfactual swap header ───────────────────────────────────────
cells.append(md("""\
## 4. Counterfactual concept swap

For an all-positive pair (species A, species B, concept C):

```
z_swapped = z_A  but  z_swapped[:, C] ← z_B[:, C]
logits_swapped = z_swapped @ W_y.T + b_y
shift_B = mean(softmax(logits_swapped)[:, B]) − mean(softmax(logits_A)[:, B])
```

- `shift_B > 0` → swapping z_C from B into A raised P(species B) → z_C carries B-identity
- `leakage_sym = 0.5 * (leakage_fwd + leakage_bwd)` — symmetric, both directions averaged"""))

# ── Cell 12: swap functions ───────────────────────────────────────────────────
cells.append(code("""\
@torch.no_grad()
def concept_swap(
    z_A: torch.Tensor,   # [N_A, K]  recipient
    z_B: torch.Tensor,   # [N_B, K]  donor
    concept_idx: int,
):
    \"\"\"
    For every A-image: replace z_A[:, concept_idx] with each of the N_B donor values,
    average the resulting softmax probabilities over donors.
    Returns p_orig [N_A, n_sp] and p_swap [N_A, n_sp].
    \"\"\"
    N_A, K = z_A.shape
    N_B    = z_B.shape[0]

    p_orig = torch.softmax(z_A @ W_y.T + b_y, dim=-1)          # [N_A, n_sp]

    z_swap = z_A.unsqueeze(1).expand(N_A, N_B, K).clone()       # [N_A, N_B, K]
    z_swap[:, :, concept_idx] = z_B[:, concept_idx].unsqueeze(0)

    logits_swap = z_swap.view(N_A * N_B, K) @ W_y.T + b_y       # [N_A*N_B, n_sp]
    p_swap = torch.softmax(logits_swap, dim=-1).view(N_A, N_B, -1).mean(dim=1)

    return p_orig, p_swap


def pair_swap_metrics(sid_A: int, sid_B: int, concept_idx: int):
    \"\"\"
    Symmetric leakage score for (species A, species B, concept C).
    leakage_sym > 0 → z_C carries species-identity signal.
    \"\"\"
    mask_A = sids_te == sid_A
    mask_B = sids_te == sid_B
    z_A    = z_te[mask_A]
    z_B    = z_te[mask_B]
    if len(z_A) == 0 or len(z_B) == 0:
        return None

    p_orig_A, p_swap_A = concept_swap(z_A, z_B, concept_idx)
    shift_B_fwd = (p_swap_A[:, sid_B] - p_orig_A[:, sid_B]).mean().item()
    shift_A_fwd = (p_swap_A[:, sid_A] - p_orig_A[:, sid_A]).mean().item()

    p_orig_B, p_swap_B = concept_swap(z_B, z_A, concept_idx)
    shift_A_bwd = (p_swap_B[:, sid_A] - p_orig_B[:, sid_A]).mean().item()
    shift_B_bwd = (p_swap_B[:, sid_B] - p_orig_B[:, sid_B]).mean().item()

    leakage_fwd = shift_B_fwd - shift_A_fwd
    leakage_bwd = shift_A_bwd - shift_B_bwd
    leakage_sym = 0.5 * (leakage_fwd + leakage_bwd)

    return {
        'sid_A': int(sid_A), 'sid_B': int(sid_B),
        'concept_idx': int(concept_idx), 'concept': CONCEPT_NAMES[concept_idx],
        'shift_B_fwd': float(shift_B_fwd),
        'shift_A_fwd': float(shift_A_fwd),
        'shift_A_bwd': float(shift_A_bwd),
        'shift_B_bwd': float(shift_B_bwd),
        'leakage_fwd': float(leakage_fwd),
        'leakage_bwd': float(leakage_bwd),
        'leakage_sym': float(leakage_sym),
        'n_A': int(mask_A.sum()),
        'n_B': int(mask_B.sum()),
    }


print('Defined: concept_swap  pair_swap_metrics')"""))

# ── Cell 13: run swaps ────────────────────────────────────────────────────────
cells.append(md("## 5. Run swaps for all GT-positive pairs"))

cells.append(code("""\
swap_rows = []

for c_idx, c_name in enumerate(CONCEPT_NAMES):
    positive_sids = np.where(cc_matrix[:, c_idx] == 1)[0]
    if len(positive_sids) < 2:
        continue
    for sid_A, sid_B in combinations(positive_sids, 2):
        r = pair_swap_metrics(sid_A, sid_B, c_idx)
        if r is not None:
            swap_rows.append(r)

swap_df = pd.DataFrame(swap_rows)
swap_df['part'] = swap_df['concept'].map(CONCEPT_TO_PART)

print(f'Swap results: {len(swap_df)} rows  ({swap_df["concept"].nunique()} concepts)')
print(f'leakage_sym  mean={swap_df["leakage_sym"].mean():.4f}  '
      f'median={swap_df["leakage_sym"].median():.4f}  '
      f'max={swap_df["leakage_sym"].max():.4f}')
swap_df.head(10)"""))

cells.append(code("""\
swap_df.to_csv('fb_cbm_counterfactual_swap.csv', index=False)
print('Saved fb_cbm_counterfactual_swap.csv')"""))

# ── Cell 14: per-concept aggregation ─────────────────────────────────────────
cells.append(md("## 6. Per-concept and per-part aggregation"))

cells.append(code("""\
concept_agg = (
    swap_df.groupby(['concept', 'part'], as_index=False)
    .agg(
        n_pairs       = ('leakage_sym', 'size'),
        leakage_mean  = ('leakage_sym', 'mean'),
        leakage_std   = ('leakage_sym', 'std'),
        leakage_max   = ('leakage_sym', 'max'),
        shift_B_mean  = ('shift_B_fwd', 'mean'),
        shift_A_mean  = ('shift_A_fwd', 'mean'),
        frac_positive = ('leakage_sym', lambda s: float((s > 0).mean())),
    )
    .sort_values('leakage_mean', ascending=False)
    .reset_index(drop=True)
)
print('Per-concept leakage summary:')
display(concept_agg)"""))

# ── Cell 15: leakage barh + per-part ─────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# Panel A: leakage_mean per concept, coloured by part
ax = axes[0]
colors = [PART_COLORS[p] for p in concept_agg['part']]
ax.barh(concept_agg['concept'], concept_agg['leakage_mean'], color=colors, alpha=0.8)
ax.errorbar(
    concept_agg['leakage_mean'], range(len(concept_agg)),
    xerr=concept_agg['leakage_std'].fillna(0),
    fmt='none', color='black', alpha=0.5, capsize=3,
)
ax.axvline(0, color='gray', ls='--', alpha=0.7)
ax.set_xlabel('Mean leakage score (shift_B − shift_A, symmetric)')
ax.set_title('Counterfactual swap leakage by concept\\n'
             '(>0 = concept activation carries species identity)')
ax.legend(handles=[Patch(color=c, label=p) for p, c in PART_COLORS.items()],
          fontsize=8, loc='lower right')
ax.grid(True, axis='x', alpha=0.3)

# Panel B: per-part aggregation
ax = axes[1]
part_agg = swap_df.groupby('part')['leakage_sym'].agg(['mean', 'sem', 'size']).reset_index()
part_agg = part_agg.sort_values('mean', ascending=False)
ax.barh(part_agg['part'],
        part_agg['mean'],
        xerr=part_agg['sem'],
        color=[PART_COLORS[p] for p in part_agg['part']],
        alpha=0.8, capsize=4)
for _, r in part_agg.iterrows():
    ax.text(max(r['mean'], 0) + 0.001, list(part_agg['part']).index(r['part']),
            f"n={int(r['size'])}", va='center', fontsize=8)
ax.axvline(0, color='gray', ls='--', alpha=0.7)
ax.set_xlabel('Mean leakage score (± SEM)')
ax.set_title('Leakage by body part')
ax.grid(True, axis='x', alpha=0.3)

plt.suptitle('FunnyBirds CBM: Counterfactual concept swap — leakage scores', y=1.02)
plt.tight_layout()
plt.savefig('fb_cbm_swap_leakage.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_cbm_swap_leakage.png')"""))

# ── Cell 16: decomposition scatter ───────────────────────────────────────────
cells.append(code("""\
# Decompose leakage: ΔP(B) vs −ΔP(A) per concept
fig, ax = plt.subplots(figsize=(7, 5))
for _, row in concept_agg.iterrows():
    color = PART_COLORS[row['part']]
    ax.scatter(row['shift_B_mean'], -row['shift_A_mean'],
               color=color, s=60, alpha=0.8, zorder=3)
    ax.annotate(row['concept'],
                (row['shift_B_mean'], -row['shift_A_mean']),
                fontsize=6, alpha=0.7)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.axvline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('ΔP(species B)  after swapping z_C from B into A  (>0 if leakage)')
ax.set_ylabel('−ΔP(species A)  (>0 if leakage)')
ax.set_title('Leakage decomposition per concept\\n'
             '(top-right = both effects consistent with leakage)')
ax.legend(handles=[Patch(color=c, label=p) for p, c in PART_COLORS.items()], fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fb_cbm_swap_decomp.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_cbm_swap_decomp.png')"""))

# ── Cell 17: species-identity probe header ────────────────────────────────────
cells.append(md("""\
## 7. Species-identity probe on z

Train a linear classifier `z [26-dim] → species_id`.
High accuracy = the 26-dimensional concept bottleneck encodes species identity, not
just concept presence.  Compare against a random-Gaussian baseline."""))

# ── Cell 18: LinearProbe using same pattern as fb_recallv2.py ─────────────────
cells.append(code("""\
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_classifier(
    X_tr, y_tr, X_te, y_te,
    n_classes: int, seed: int = 0,
    lr: float = 1e-2, wd: float = 1e-4,
    epochs: int = 30, batch: int = 512,
) -> float:
    \"\"\"Train linear multiclass probe; return test accuracy.\"\"\"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    X_tr = X_tr.to(device); X_te = X_te.to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    probe   = LinearProbe(X_tr.shape[1], n_classes).to(device)
    opt     = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    n = X_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx  = perm[i:i+batch]
            loss = loss_fn(probe(X_tr[idx]), y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        probe.eval()
        preds = probe(X_te).argmax(dim=1)
        return float((preds == y_te_t).float().mean())


print('Defined: LinearProbe  train_classifier')"""))

# ── Cell 19: run species probe ────────────────────────────────────────────────
cells.append(code("""\
# Full z (26-dim) → species probe
acc_full = train_classifier(
    z_tr, sids_tr, z_te, sids_te, n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
)
print(f'Species probe on z (all 26 concepts): {acc_full:.4f}')
print(f'  chance level: {1/N_SPECIES:.4f}  ({N_SPECIES} classes)')

# Random baseline: same shape, Gaussian noise
torch.manual_seed(42)
acc_rand = train_classifier(
    torch.randn_like(z_tr), sids_tr,
    torch.randn_like(z_te), sids_te,
    n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
)
print(f'Species probe on random baseline:    {acc_rand:.4f}')

# avgpool backbone baseline: how much species info before the bottleneck?
acc_avgpool = train_classifier(
    avg_tr, sids_tr, avg_te, sids_te, n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
)
print(f'Species probe on avgpool (backbone): {acc_avgpool:.4f}')"""))

# ── Cell 20: per-concept species probe ───────────────────────────────────────
cells.append(code("""\
# Per-concept probe: z[:, c] (single scalar) → species_id
per_concept_acc = {}
for c_idx, c_name in enumerate(CONCEPT_NAMES):
    acc = train_classifier(
        z_tr[:, c_idx:c_idx+1], sids_tr,
        z_te[:, c_idx:c_idx+1], sids_te,
        n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
    )
    per_concept_acc[c_name] = acc

probe_acc_df = pd.DataFrame([
    {'concept': c, 'species_acc': a, 'part': CONCEPT_TO_PART[c]}
    for c, a in per_concept_acc.items()
]).sort_values('species_acc', ascending=False)

print(f'\\nPer-concept species decodability (chance={1/N_SPECIES:.3f}):')
display(probe_acc_df)"""))

# ── Cell 21: species probe plots ─────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# Panel A: per-concept species probe accuracy
ax = axes[0]
colors = [PART_COLORS[p] for p in probe_acc_df['part']]
ax.barh(probe_acc_df['concept'], probe_acc_df['species_acc'], color=colors, alpha=0.8)
ax.axvline(1/N_SPECIES, color='red', ls='--', alpha=0.7, label=f'chance 1/{N_SPECIES}')
ax.axvline(acc_full,    color='black', ls='-', alpha=0.5, label=f'all-z ({acc_full:.3f})')
ax.set_xlabel('Linear probe accuracy: z_C → species_id')
ax.set_title('Species-identity decodability per concept activation\\n'
             '(above chance = concept bottleneck leaks species identity)')
ax.legend(handles=[
    *[Patch(color=c, label=p) for p, c in PART_COLORS.items()],
    Line2D([0],[0], color='red',   ls='--', label=f'chance 1/{N_SPECIES}'),
    Line2D([0],[0], color='black',          label=f'all-z ({acc_full:.3f})'),
], fontsize=7)
ax.grid(True, axis='x', alpha=0.3)

# Panel B: summary bar — backbone avgpool vs z vs random
ax = axes[1]
labels = ['Random\\n(baseline)', 'CBM z\\n(all 26)', 'avgpool\\n(backbone)']
accs   = [acc_rand, acc_full, acc_avgpool]
ax.bar(labels, accs, color=['lightgray', 'steelblue', 'darkorange'], alpha=0.85)
ax.axhline(1/N_SPECIES, color='red', ls='--', alpha=0.7, label=f'chance 1/{N_SPECIES}')
for i, a in enumerate(accs):
    ax.text(i, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
ax.set_ylabel('Species prediction accuracy')
ax.set_title('How much species identity is in each representation?')
ax.legend(); ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, min(1.0, max(accs) * 1.2))

plt.suptitle('FunnyBirds CBM: Species-identity decodability from concept bottleneck', y=1.02)
plt.tight_layout()
plt.savefig('fb_cbm_species_probe.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_cbm_species_probe.png')"""))

# ── Cell 22: binary discriminability header ───────────────────────────────────
cells.append(md("""\
## 8. Per-pair binary discriminability in z_C

For each all-positive pair (A, B, concept C): train `z_C [scalar] → {species A, species B}`.
Accuracy > 0.5 = z_C distinguishes the two species — mechanistic evidence that the concept
activation carries species-specific signal beyond whether the concept is present."""))

# ── Cell 23: binary disc function ────────────────────────────────────────────
cells.append(code("""\
def binary_species_disc(sid_A: int, sid_B: int, concept_idx: int) -> float:
    \"\"\"
    Train z[:, concept_idx] → {0=A, 1=B} binary probe.
    Returns test accuracy (0.5 = chance).  NaN if not enough data.
    \"\"\"
    mask_tr = np.isin(sids_tr, [sid_A, sid_B])
    mask_te = np.isin(sids_te, [sid_A, sid_B])
    if mask_tr.sum() < 4 or mask_te.sum() < 2:
        return float('nan')
    X_tr = z_tr[mask_tr, concept_idx:concept_idx+1]
    X_te = z_te[mask_te, concept_idx:concept_idx+1]
    y_tr = (sids_tr[mask_tr] == sid_B).astype(int)
    y_te = (sids_te[mask_te] == sid_B).astype(int)
    return train_classifier(X_tr, y_tr, X_te, y_te, n_classes=2, epochs=PROBE_EPOCHS)


print('Defined: binary_species_disc')"""))

# ── Cell 24: run binary disc ──────────────────────────────────────────────────
cells.append(code("""\
disc_rows = []
for c_idx, c_name in enumerate(CONCEPT_NAMES):
    positive_sids = np.where(cc_matrix[:, c_idx] == 1)[0]
    if len(positive_sids) < 2:
        continue
    for sid_A, sid_B in combinations(positive_sids, 2):
        acc = binary_species_disc(sid_A, sid_B, c_idx)
        if not np.isnan(acc):
            disc_rows.append({
                'sid_A': int(sid_A), 'sid_B': int(sid_B),
                'concept': c_name, 'concept_idx': c_idx,
                'part': CONCEPT_TO_PART[c_name],
                'disc_acc': float(acc),
            })

disc_df = pd.DataFrame(disc_rows)
disc_df['disc_above_chance'] = disc_df['disc_acc'] > 0.5

print(f'Binary discriminability: {len(disc_df)} pair-concept combos')
print(f'  mean disc_acc={disc_df["disc_acc"].mean():.4f}  '
      f'frac>0.5: {disc_df["disc_above_chance"].mean():.3f}')

disc_concept = (
    disc_df.groupby(['concept', 'part'])[['disc_acc', 'disc_above_chance']]
    .mean().reset_index()
    .sort_values('disc_acc', ascending=False)
)
display(disc_concept)"""))

# ── Cell 25: binary disc plots ────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

ax = axes[0]
colors = [PART_COLORS[p] for p in disc_concept['part']]
ax.barh(disc_concept['concept'], disc_concept['disc_acc'], color=colors, alpha=0.8)
ax.axvline(0.5, color='red', ls='--', alpha=0.7, label='chance (0.5)')
ax.set_xlabel('Mean binary accuracy: z_C → {species A, species B}')
ax.set_title('Per-pair species discriminability in z_C\\n'
             '(averaged over all GT-positive pairs for each concept)')
ax.legend(handles=[
    *[Patch(color=c, label=p) for p, c in PART_COLORS.items()],
    Line2D([0],[0], color='red', ls='--', label='chance 0.5'),
], fontsize=7)
ax.grid(True, axis='x', alpha=0.3)

ax = axes[1]
part_disc = (
    disc_df.groupby('part')['disc_acc']
    .agg(['mean', 'sem'])
    .reset_index()
    .sort_values('mean', ascending=False)
)
ax.barh(part_disc['part'],
        part_disc['mean'],
        xerr=part_disc['sem'],
        color=[PART_COLORS[p] for p in part_disc['part']],
        alpha=0.8, capsize=4)
ax.axvline(0.5, color='red', ls='--', alpha=0.7, label='chance 0.5')
ax.set_xlabel('Mean discriminability (± SEM)')
ax.set_title('Discriminability by body part')
ax.legend(); ax.grid(True, axis='x', alpha=0.3)

plt.suptitle('FunnyBirds CBM: Species discriminability in z_C (per-concept scalar)', y=1.02)
plt.tight_layout()
plt.savefig('fb_cbm_binary_disc.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_cbm_binary_disc.png')"""))

# ── Cell 26: three-level evidence ────────────────────────────────────────────
cells.append(md("""\
## 9. Three-level evidence: correlation across measures

| Analysis | Level | Measure |
|---|---|---|
| Recall gap (`fb_recallv2.ipynb`) | Behavioural | Probe fires differently across species |
| Counterfactual swap | Causal | Replacing z_C shifts species prediction |
| Binary discriminability | Mechanistic | z_C alone separates two species |

Per-concept correlation across the three measures — if they agree, the evidence
for species-identity leakage through the concept bottleneck is robust."""))

cells.append(code("""\
# Merge the three concept-level measures
merged = (
    concept_agg[['concept', 'part', 'leakage_mean']]
    .merge(disc_concept[['concept', 'disc_acc']], on='concept', how='inner')
    .merge(probe_acc_df[['concept', 'species_acc']], on='concept', how='inner')
)

# Optionally load recall gap from fb_recallv2 output
CBM_SPECIES_CSV = Path('fb_cbm_species.csv')
if CBM_SPECIES_CSV.exists():
    sp_csv = pd.read_csv(CBM_SPECIES_CSV)
    if {'attr', 'recall'}.issubset(sp_csv.columns):
        recall_gap = (
            sp_csv.groupby('attr')['recall']
            .apply(lambda x: x.max() - x.min())
            .reset_index(name='recall_range')
            .rename(columns={'attr': 'concept'})
        )
        merged = merged.merge(recall_gap, on='concept', how='left')
        print('Merged recall_range from fb_cbm_species.csv')
    else:
        merged['recall_range'] = np.nan
else:
    merged['recall_range'] = np.nan
    print('[info] fb_cbm_species.csv not found — run fb_recallv2.ipynb first')

display(merged)"""))

cells.append(code("""\
has_recall = merged['recall_range'].notna().any()
ncols = 3 if has_recall else 2
fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))

pairs_to_plot = [
    ('leakage_mean', 'disc_acc',    'Leakage score vs Discriminability'),
    ('leakage_mean', 'species_acc', 'Leakage score vs Species probe acc'),
]
if has_recall:
    pairs_to_plot.append(('recall_range', 'leakage_mean', 'Recall range vs Leakage score'))

for ax, (x_col, y_col, title) in zip(axes, pairs_to_plot):
    sub = merged[[x_col, y_col, 'concept', 'part']].dropna()
    colors_sub = [PART_COLORS[p] for p in sub['part']]
    ax.scatter(sub[x_col], sub[y_col], c=colors_sub, s=60, alpha=0.8, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row['concept'], (row[x_col], row[y_col]), fontsize=6, alpha=0.6)
    if len(sub) > 2:
        r, p = stats.pearsonr(sub[x_col], sub[y_col])
        m, b = np.polyfit(sub[x_col], sub[y_col], 1)
        xs   = np.array([sub[x_col].min(), sub[x_col].max()])
        ax.plot(xs, m * xs + b, 'k--', alpha=0.5, label=f'r={r:.2f}, p={p:.3f}')
        ax.legend(fontsize=8)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title); ax.grid(True, alpha=0.3)

plt.suptitle('FunnyBirds CBM: Cross-measure correlation (three levels of evidence)', y=1.02)
plt.tight_layout()
plt.savefig('fb_cbm_evidence_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_cbm_evidence_correlation.png')"""))

# ── Cell 27: summary ─────────────────────────────────────────────────────────
cells.append(md("""\
## 10. Summary

### What we measured

| Measure | Value | Interpretation |
|---|---|---|
| Species probe on z (all 26) | `acc_full` | How much species info survives the bottleneck |
| Species probe on avgpool (backbone) | `acc_avgpool` | Upstream ceiling |
| Mean leakage score (swap) | `leakage_sym` | Causal: z_C carries species signal |
| Mean binary discriminability | `disc_acc` | z_C scalar separates species |

### Interpreting leakage_sym

`leakage_sym > 0` for a concept means: swapping z_C from species B into species A
consistently raised P(B) and lowered P(A).  The bottleneck value is not purely
concept-predictive — it also carries which species produced it.

A concept with **high leakage + high discriminability + high species probe acc**
is the strongest case: all three independent analyses agree it leaks species identity.

### Next: MCBM comparison

Run the MCBM version of this notebook (to be created) across γ values.
The IB penalty should reduce `leakage_sym` and `disc_acc` while preserving
concept accuracy — that is the direct mechanistic test of the backwash hypothesis."""))

# ── Build and write notebook ──────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("funnybird_notebooks/fb_cbm_counterfactual.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")

nb2 = json.loads(out.read_text())
code_cells = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
md_cells   = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
print(f"Cells: {len(nb2['cells'])} total ({code_cells} code, {md_cells} markdown)")
print("Valid JSON: OK")
