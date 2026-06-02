"""
Generator script: produces notebooks/cub_cbm_counterfactual.ipynb
Mirrors funnybird_notebooks/fb_cbm_counterfactual.ipynb cell-by-cell for CUB-200-2011.

Key differences vs FunnyBirds:
  - 200 species, ~100 attribute concepts (vs 50 species, 26 binary concepts)
  - Concept activations: z = avgpool @ W_c.T + b_c  (NO sigmoid; MCBM concept_encoder)
  - Species IDs are 1-indexed (1-200); subtract _sid_min for label-head indexing
  - class-concept matrix built from training data (majority vote per species)
  - Attribute groups from attributes.txt (has_bill_color, etc.)
  - MAX_PAIRS_PER_CONCEPT limits for tractable runtime
"""
import json, textwrap
from pathlib import Path

def code(src): return {"cell_type":"code","metadata":{},"source":[src],"outputs":[],"execution_count":None}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":[src]}

cells = []

# ── 0: Title ─────────────────────────────────────────────────────────────────
cells.append(md("""# CUB-200-2011 CBM — Counterfactual Concept Swap

**Direct parallel to `fb_cbm_counterfactual.ipynb`** — same three analyses on CUB.

Key differences vs FunnyBirds:
- 200 species, ~100 attribute concepts (vs 50 / 26)
- Concept activations: `z = avgpool @ W_c.T + b_c` — **no sigmoid** (MCBM concept_encoder)
- Species IDs are 1-indexed (1–200); `_sid_min` offset for label-head indexing
- Class-concept matrix built from training-set majority vote (not exact GT)
- Attribute groups from `attributes.txt` (e.g. `has_bill_color`, `has_wing_color`)
- `MAX_PAIRS_PER_CONCEPT` cap for tractable runtime

**Three analyses:**
1. **Counterfactual swap** — replace z_A[C] with z_B[C] for all-positive pair (A,B,C); measure ΔP(B)
2. **Species-identity probe on z** — linear z→species; how much species info survives the bottleneck
3. **Binary discriminability** — z_C scalar → {A,B} binary probe per all-positive pair

**New diagnostics (cells 38–41):**
- Leakage distribution & per-species heterogeneity
- Between/within-species variance ratio for z_C
- Per-pair discriminability vs leakage scatter

Data-loading follows `recall.ipynb` exactly."""))

# ── 1: Imports ────────────────────────────────────────────────────────────────
cells.append(code("""import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats"""))

# ── 2: Paths md ───────────────────────────────────────────────────────────────
cells.append(md("## 0. Paths and configuration"))

# ── 3: Paths ──────────────────────────────────────────────────────────────────
cells.append(code("""import sys
ROOT = Path('/scratch/network/cr7998/cv_emergence_project')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUB       = ROOT / 'data' / 'CUB_200_2011'
ATTR_TXT  = ROOT / 'data' / 'attributes.txt'   # attr_id -> attr_name like has_bill_color::black

# MCBM gamma=0.0 IS the standard CBM (no IB penalty)
CBM_CKPT  = ROOT / 'checkpoints_mcbm' / 'mcbm_gamma0.0.pth'
CBM_FEATS = ROOT / 'features' / 'resnet50_mcbm_gamma0.0'

assert CUB.exists(),       f'Missing: {CUB}'
assert ATTR_TXT.exists(),  f'Missing: {ATTR_TXT}'
assert CBM_CKPT.exists(),  f'Missing checkpoint: {CBM_CKPT}'
assert CBM_FEATS.exists(), f'Missing features:   {CBM_FEATS}'

N_SPECIES    = 200
PROBE_EPOCHS = 30
MAX_PAIRS_PER_CONCEPT = 50   # cap for tractable runtime (all-positive pairs can be huge for CUB)
MAX_DISC_PAIRS        = 500  # cap for discriminability loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[config] device: {device}')
print(f'[config] CBM checkpoint: {CBM_CKPT}')"""))

# ── 4: helpers md ────────────────────────────────────────────────────────────
cells.append(md("## 1. Data-loading helpers  *(from recall.ipynb)*"))

# ── 5: load helpers ──────────────────────────────────────────────────────────
cells.append(code("""def load_species_maps(cub_root: Path):
    \"\"\"Load species ID -> name from classes.txt (1-indexed, 1..200).\"\"\"
    classes = pd.read_csv(
        cub_root / 'classes.txt',
        sep=r'\\s+', header=None,
        names=['species_id', 'class_name'], engine='python',
    )
    def pretty(name): return name.split('.', 1)[-1].replace('_', ' ')
    return {int(r.species_id): pretty(r.class_name) for _, r in classes.iterrows()}

species_id_to_name = load_species_maps(CUB)
def spname(sid): return species_id_to_name.get(int(sid), f'species_{sid}')


def load_meta(cub_root: Path) -> pd.DataFrame:
    img_sp = pd.read_csv(
        cub_root / 'image_class_labels.txt',
        sep=r'\\s+', header=None, names=['image_id','species_id'], engine='python',
    )
    split = pd.read_csv(
        cub_root / 'train_test_split.txt',
        sep=r'\\s+', header=None, names=['image_id','is_train'], engine='python',
    )
    meta = img_sp.merge(split, on='image_id')
    meta['species_name'] = meta['species_id'].map(spname)
    return meta


def load_image_attr_labels_robust(cub_root: Path) -> pd.DataFrame:
    path = cub_root / 'attributes' / 'image_attribute_labels.txt'
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 4:
                bad += 1; continue
            try:
                rows.append((int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])))
            except:
                bad += 1
    df = pd.DataFrame(rows, columns=['image_id','attr_id','is_present','certainty'])
    print(f'Parsed rows: {len(df)}  bad lines: {bad}')
    return df


def load_attr_maps(attr_txt: Path):
    rows = []
    with open(attr_txt) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            aid_str, name = line.split(' ', 1)
            rows.append((int(aid_str), name))
    df = pd.DataFrame(rows, columns=['attr_id','attr_name'])
    name2id = dict(zip(df['attr_name'], df['attr_id']))
    id2name  = dict(zip(df['attr_id'],  df['attr_name']))
    return df, name2id, id2name


def build_attr_labeled_df(meta, img_attr_long, attr_id, min_certainty=1):
    sub = img_attr_long[img_attr_long['attr_id'] == int(attr_id)].copy()
    sub = sub[sub['certainty'] >= min_certainty].copy()
    out = meta.merge(sub[['image_id','is_present','certainty']], on='image_id', how='inner')
    out = out.rename(columns={'is_present':'y'})
    out['y'] = out['y'].astype(int)
    return out[['image_id','species_id','species_name','is_train','y','certainty']]


print('Defined: load_species_maps  load_meta  load_image_attr_labels_robust')
print('         load_attr_maps  build_attr_labeled_df')"""))

# ── 6: torch helpers ─────────────────────────────────────────────────────────
cells.append(code("""def safe_torch_load(path: Path):
    try:    return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError: return torch.load(path, map_location='cpu')

def load_features(feat_dir: Path, layer: str, split: str) -> torch.Tensor:
    p = feat_dir / f'{layer}_{split}.pt'
    assert p.exists(), f'Missing: {p}'
    X = safe_torch_load(p)
    if not isinstance(X, torch.Tensor): X = torch.tensor(X)
    return X.float()

def to_1d_int_array(x):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    return np.array(x).reshape(-1).astype(int)

def infer_kind(arr):
    if arr.max() <= 200 and arr.min() >= 0: return 'species_id_like'
    if arr.max() > 200:                      return 'image_id_like'
    return 'unknown'

def load_split_order(feat_dir: Path, split: str):
    p = feat_dir / f'labels_{split}.pt'
    assert p.exists(), f'Missing: {p}'
    t = safe_torch_load(p)
    assert isinstance(t, dict) and 'image_ids' in t, f'{p} missing image_ids'
    ids = to_1d_int_array(t['image_ids'])
    return infer_kind(ids), ids

def align_features_and_labels(X_split, image_ids_in_feature_order, labeled_df_split):
    labeled = labeled_df_split.set_index('image_id')[['y','species_id','species_name']]
    keep_idx, rows = [], []
    for i, img_id in enumerate(image_ids_in_feature_order):
        img_id = int(img_id)
        if img_id in labeled.index:
            keep_idx.append(i)
            y, sid, sname = labeled.loc[img_id]
            rows.append((img_id, int(sid), str(sname), int(y)))
    X_aligned  = X_split[keep_idx]
    df_aligned = pd.DataFrame(rows, columns=['image_id','species_id','species_name','y'])
    return X_aligned, df_aligned

print('Defined: safe_torch_load  load_features  load_split_order')
print('         to_1d_int_array  infer_kind  align_features_and_labels')"""))

# ── 7: Concept names md ───────────────────────────────────────────────────────
cells.append(md("""## 2. Concept names, attribute groups, class-concept matrix

`chosen_attr_cols` and `num_concepts` are saved in the MCBM checkpoint config.
Attribute groups come from the `::` split of the attribute name
(e.g. `has_bill_color::black` → group `has_bill_color`)."""))

# ── 8: Load checkpoint config ─────────────────────────────────────────────────
cells.append(code("""ckpt_meta = safe_torch_load(CBM_CKPT)
cfg = ckpt_meta.get('config', {})
chosen_attr_cols = cfg.get('attr_cols', [])   # list of 'attr_NNN_present' strings
N_CONCEPTS = int(cfg.get('num_concepts', len(chosen_attr_cols)))
print(f'MCBM config: num_concepts={N_CONCEPTS}')
print(f'attr_cols[:5]: {chosen_attr_cols[:5]}')"""))

# ── 9: Concept names + group colors ──────────────────────────────────────────
cells.append(code("""attr_df_full, attr_name_to_id, attr_id_to_name = load_attr_maps(ATTR_TXT)

# Map chosen attr_cols -> concept names and groups
CONCEPT_NAMES = []
ATTR_TO_GROUP = {}
for col in chosen_attr_cols:
    attr_id   = int(col.split('_')[1])   # 'attr_10_present' -> 10
    attr_name = attr_id_to_name[attr_id]  # e.g. 'has_bill_color::black'
    group     = attr_name.split('::')[0]  # e.g. 'has_bill_color'
    CONCEPT_NAMES.append(attr_name)
    ATTR_TO_GROUP[attr_name] = group

CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPT_NAMES)}

# Build group -> color mapping using tab20 colormap
unique_groups = sorted(set(ATTR_TO_GROUP.values()))
_cmap = plt.cm.get_cmap('tab20', len(unique_groups))
GROUP_COLORS = {g: _cmap(i) for i, g in enumerate(unique_groups)}
CONCEPT_TO_COLOR = {c: GROUP_COLORS[ATTR_TO_GROUP[c]] for c in CONCEPT_NAMES}

# PART_GROUPS equivalent (group -> list of concept indices)
PART_GROUPS = {}
for i, c in enumerate(CONCEPT_NAMES):
    g = ATTR_TO_GROUP[c]
    PART_GROUPS.setdefault(g, []).append(i)

print(f'N_CONCEPTS: {N_CONCEPTS}  unique groups: {len(unique_groups)}')
print(f'Groups: {unique_groups[:10]} ...')"""))

# ── 10: Load metadata + cc_matrix ─────────────────────────────────────────────
cells.append(code("""meta         = load_meta(CUB)
img_attr_long = load_image_attr_labels_robust(CUB)

_sid_min = int(meta['species_id'].min())   # 1 for CUB; used to 0-index label head
print(f'Species ID range: {_sid_min} .. {meta["species_id"].max()}  (_sid_min={_sid_min})')
print(f'meta: {len(meta)} images  (train={meta.is_train.sum()}, test={(meta.is_train==0).sum()})')

# Build cc_matrix [N_SPECIES, N_CONCEPTS] from training-data majority vote
# cc_matrix[s_0idx, c] = 1  if  mean(is_present) >= 0.5  for species s, attr c in train split
print('Building class-concept matrix from training data...')
_meta_tr = meta[meta['is_train'] == 1][['image_id','species_id']].copy()

cc_matrix = np.zeros((N_SPECIES, N_CONCEPTS), dtype=int)
for c_idx, c_name in enumerate(CONCEPT_NAMES):
    attr_id = attr_name_to_id[c_name]
    sub = (img_attr_long[img_attr_long['attr_id'] == attr_id]
           .merge(_meta_tr, on='image_id', how='inner'))
    sp_mean = sub.groupby('species_id')['is_present'].mean()
    for sid, val in sp_mean.items():
        s_0idx = int(sid) - _sid_min
        if 0 <= s_0idx < N_SPECIES:
            cc_matrix[s_0idx, c_idx] = int(val >= 0.5)

pos_per_concept = cc_matrix.sum(axis=0)
print(f'cc_matrix: {cc_matrix.shape}  mean positive species per concept: {pos_per_concept.mean():.1f}')"""))

# ── 11: Load split ordering ───────────────────────────────────────────────────
cells.append(code("""cbm_kind_tr, cbm_ids_tr = load_split_order(CBM_FEATS, 'train')
cbm_kind_te, cbm_ids_te = load_split_order(CBM_FEATS, 'test')

print(f'CBM train: {cbm_kind_tr}  id range: ({cbm_ids_tr.min()}, {cbm_ids_tr.max()})')
print(f'CBM test:  {cbm_kind_te}  id range: ({cbm_ids_te.min()}, {cbm_ids_te.max()})')
print(f'Train N={len(cbm_ids_tr)}  Test N={len(cbm_ids_te)}')"""))

# ── 12: Load CBM weights md ───────────────────────────────────────────────────
cells.append(md("""## 3. Load CBM weights and compute concept activations

Architecture (MCBM gamma=0 = standard CBM):
`backbone → avgpool [2048] → concept_encoder [K] → label_head [200]`

- `concept_encoder`: `W_c [K, 2048]`, `b_c [K]`  → `z = avgpool @ W_c.T + b_c`
- `label_head`:      `W_y [200, K]`,  `b_y [200]`  → `logits = z @ W_y.T + b_y`

**No sigmoid** between concept_encoder and label_head (unlike FunnyBirds CBM).
`sigmoid(z)` used only for concept presence prediction, not as input to label_head."""))

# ── 13: Load weights ──────────────────────────────────────────────────────────
cells.append(code("""sd = ckpt_meta.get('model_state_dict', ckpt_meta)

W_c = sd['concept_encoder.weight'].float()   # [K, 2048]
b_c = sd['concept_encoder.bias'].float()     # [K]
W_y = sd['label_head.weight'].float()        # [200, K]
b_y = sd['label_head.bias'].float()          # [200]

print(f'concept_encoder: {tuple(W_c.shape)}  bias: {tuple(b_c.shape)}')
print(f'label_head:      {tuple(W_y.shape)}  bias: {tuple(b_y.shape)}')
assert W_c.shape[0] == N_CONCEPTS, f'K mismatch: {W_c.shape[0]} vs {N_CONCEPTS}'"""))

# ── 14: Compute z ─────────────────────────────────────────────────────────────
cells.append(code("""avg_te = load_features(CBM_FEATS, 'avgpool', 'test')
avg_tr = load_features(CBM_FEATS, 'avgpool', 'train')

_meta_idx = meta.set_index('image_id')
sids_te = np.array([int(_meta_idx.loc[int(i), 'species_id']) for i in cbm_ids_te])
sids_tr = np.array([int(_meta_idx.loc[int(i), 'species_id']) for i in cbm_ids_tr])

@torch.no_grad()
def compute_z(avgpool: torch.Tensor) -> torch.Tensor:
    \"\"\"z = avgpool @ W_c.T + b_c  [N, K]  (raw logits, NO sigmoid)\"\"\"
    return avgpool @ W_c.T + b_c

@torch.no_grad()
def compute_logits(z: torch.Tensor) -> torch.Tensor:
    return z @ W_y.T + b_y

z_te = compute_z(avg_te)   # [N_te, K]
z_tr = compute_z(avg_tr)   # [N_tr, K]

print(f'avgpool_te: {tuple(avg_te.shape)}  ->  z_te: {tuple(z_te.shape)}')
print(f'avgpool_tr: {tuple(avg_tr.shape)}  ->  z_tr: {tuple(z_tr.shape)}')"""))

# ── 15: Sanity ────────────────────────────────────────────────────────────────
cells.append(code("""# Species accuracy
logits_te = compute_logits(z_te)
pred_sp   = logits_te.argmax(dim=1).numpy()
sids_te_0 = sids_te - _sid_min   # 0-indexed for label-head comparison
sp_acc    = float((pred_sp == sids_te_0).mean())
print(f'CBM test species accuracy: {sp_acc:.4f}')
print(f'  chance: {1/N_SPECIES:.4f}  ({N_SPECIES} classes)')

# Concept accuracy vs GT (majority-vote cc_matrix)
gt_concepts   = cc_matrix[sids_te_0]                    # [N_te, K]
pred_concepts = (torch.sigmoid(z_te).numpy() > 0.5).astype(int)
print(f'Concept accuracy (sigmoid>0.5 vs training majority): '
      f'{(pred_concepts == gt_concepts).mean():.4f}')"""))

# ── 16: Swap md ───────────────────────────────────────────────────────────────
cells.append(md("""## 4. Counterfactual concept swap

For an all-positive pair (species A, species B, concept C):
```
z_swapped = z_A  but  z_swapped[:, C] <- z_B[:, C]
logits = z_swapped @ W_y.T + b_y
shift_B = mean(softmax(logits_swapped)[:, B_0idx]) - mean(softmax(logits_A)[:, B_0idx])
```
Species IDs are 0-indexed for label-head access: `idx = sid - _sid_min`

`leakage_sym = 0.5*(leakage_fwd + leakage_bwd)` — symmetric average."""))

# ── 17: swap functions ────────────────────────────────────────────────────────
cells.append(code("""@torch.no_grad()
def concept_swap(z_A, z_B, concept_idx):
    \"\"\"Replace z_A[:, concept_idx] with z_B[:, concept_idx], average over donors.\"\"\"
    N_A, K = z_A.shape
    N_B    = z_B.shape[0]
    p_orig = torch.softmax(z_A @ W_y.T + b_y, dim=-1)
    z_swap = z_A.unsqueeze(1).expand(N_A, N_B, K).clone()
    z_swap[:, :, concept_idx] = z_B[:, concept_idx].unsqueeze(0)
    logits_swap = z_swap.view(N_A * N_B, K) @ W_y.T + b_y
    p_swap = torch.softmax(logits_swap, dim=-1).view(N_A, N_B, -1).mean(dim=1)
    return p_orig, p_swap


def pair_swap_metrics(sid_A, sid_B, concept_idx):
    \"\"\"
    Symmetric leakage score for (species A, species B, concept C).
    Uses 0-indexed species for label-head access.
    \"\"\"
    idx_A = int(sid_A) - _sid_min
    idx_B = int(sid_B) - _sid_min
    mask_A = sids_te == sid_A
    mask_B = sids_te == sid_B
    z_A = z_te[mask_A]
    z_B = z_te[mask_B]
    if len(z_A) == 0 or len(z_B) == 0:
        return None

    p_orig_A, p_swap_A = concept_swap(z_A, z_B, concept_idx)
    shift_B_fwd = (p_swap_A[:, idx_B] - p_orig_A[:, idx_B]).mean().item()
    shift_A_fwd = (p_swap_A[:, idx_A] - p_orig_A[:, idx_A]).mean().item()

    p_orig_B, p_swap_B = concept_swap(z_B, z_A, concept_idx)
    shift_A_bwd = (p_swap_B[:, idx_A] - p_orig_B[:, idx_A]).mean().item()
    shift_B_bwd = (p_swap_B[:, idx_B] - p_orig_B[:, idx_B]).mean().item()

    leakage_fwd = shift_B_fwd - shift_A_fwd
    leakage_bwd = shift_A_bwd - shift_B_bwd
    leakage_sym = 0.5 * (leakage_fwd + leakage_bwd)

    return {
        'sid_A': int(sid_A), 'sid_B': int(sid_B),
        'concept_idx': int(concept_idx), 'concept': CONCEPT_NAMES[concept_idx],
        'group': ATTR_TO_GROUP[CONCEPT_NAMES[concept_idx]],
        'shift_B_fwd': float(shift_B_fwd), 'shift_A_fwd': float(shift_A_fwd),
        'shift_A_bwd': float(shift_A_bwd), 'shift_B_bwd': float(shift_B_bwd),
        'leakage_fwd': float(leakage_fwd), 'leakage_bwd': float(leakage_bwd),
        'leakage_sym': float(leakage_sym),
        'n_A': int(mask_A.sum()), 'n_B': int(mask_B.sum()),
    }

print('Defined: concept_swap  pair_swap_metrics')"""))

# ── 18: Run swaps md ──────────────────────────────────────────────────────────
cells.append(md("""## 5. Run swaps for all GT-positive pairs

`MAX_PAIRS_PER_CONCEPT` caps the number of pairs per concept to keep runtime tractable.
Pairs are sampled randomly (fixed seed) when the full set exceeds the cap."""))

# ── 19: Run swaps ─────────────────────────────────────────────────────────────
cells.append(code("""swap_rows = []
rng_pairs = np.random.default_rng(42)

for c_idx, c_name in enumerate(CONCEPT_NAMES):
    pos_sids = np.where(cc_matrix[:, c_idx] == 1)[0] + _sid_min  # back to 1-indexed
    if len(pos_sids) < 2:
        continue
    all_pairs = list(combinations(pos_sids, 2))
    if len(all_pairs) > MAX_PAIRS_PER_CONCEPT:
        idx = rng_pairs.choice(len(all_pairs), MAX_PAIRS_PER_CONCEPT, replace=False)
        all_pairs = [all_pairs[i] for i in idx]

    for sid_A, sid_B in all_pairs:
        r = pair_swap_metrics(sid_A, sid_B, c_idx)
        if r is not None:
            swap_rows.append(r)

swap_df = pd.DataFrame(swap_rows)
print(f'Swap results: {len(swap_df)} rows  ({swap_df["concept"].nunique()} concepts)')
print(f'leakage_sym  mean={swap_df["leakage_sym"].mean():.4f}  '
      f'median={swap_df["leakage_sym"].median():.4f}  '
      f'max={swap_df["leakage_sym"].max():.4f}')
swap_df.head(10)"""))

# ── 20: Save ──────────────────────────────────────────────────────────────────
cells.append(code("""swap_df.to_csv('cub_cbm_counterfactual_swap.csv', index=False)
print('Saved cub_cbm_counterfactual_swap.csv')"""))

# ── 21: Aggregation md ───────────────────────────────────────────────────────
cells.append(md("## 6. Per-concept and per-group aggregation"))

# ── 22: concept_agg ───────────────────────────────────────────────────────────
cells.append(code("""concept_agg = (
    swap_df.groupby(['concept','group'], as_index=False)
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
display(concept_agg.head(20))"""))

# ── 23: Leakage plot ──────────────────────────────────────────────────────────
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(concept_agg)*0.2)))

# Panel A: leakage_mean per concept
ax = axes[0]
colors = [CONCEPT_TO_COLOR[c] for c in concept_agg['concept']]
ax.barh(concept_agg['concept'], concept_agg['leakage_mean'], color=colors, alpha=0.8)
ax.errorbar(
    concept_agg['leakage_mean'], range(len(concept_agg)),
    xerr=concept_agg['leakage_std'].fillna(0),
    fmt='none', color='black', alpha=0.5, capsize=2,
)
ax.axvline(0, color='gray', ls='--', alpha=0.7)
ax.set_xlabel('Mean leakage score (shift_B − shift_A, symmetric)')
ax.set_title('Counterfactual swap leakage by concept\\n'
             '(>0 = concept activation carries species identity)')
ax.tick_params(axis='y', labelsize=6)
ax.grid(True, axis='x', alpha=0.3)

# Panel B: per-group aggregation
ax = axes[1]
group_agg = swap_df.groupby('group')['leakage_sym'].agg(['mean','sem','size']).reset_index()
group_agg = group_agg.sort_values('mean', ascending=False)
ax.barh(group_agg['group'],
        group_agg['mean'],
        xerr=group_agg['sem'],
        color=[GROUP_COLORS[g] for g in group_agg['group']],
        alpha=0.8, capsize=4)
for _, r in group_agg.iterrows():
    ax.text(max(r['mean'], 0) + 0.0005,
            list(group_agg['group']).index(r['group']),
            f"n={int(r['size'])}", va='center', fontsize=7)
ax.axvline(0, color='gray', ls='--', alpha=0.7)
ax.set_xlabel('Mean leakage score (± SEM)')
ax.set_title('Leakage by attribute group')
ax.grid(True, axis='x', alpha=0.3)

plt.suptitle('CUB CBM: Counterfactual concept swap — leakage scores', y=1.01)
plt.tight_layout()
plt.savefig('cub_cbm_swap_leakage.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved cub_cbm_swap_leakage.png')"""))

# ── 24: Decomposition ─────────────────────────────────────────────────────────
cells.append(code("""fig, ax = plt.subplots(figsize=(8, 6))
for _, row in concept_agg.iterrows():
    ax.scatter(row['shift_B_mean'], -row['shift_A_mean'],
               color=CONCEPT_TO_COLOR[row['concept']], s=30, alpha=0.7, zorder=3)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.axvline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('ΔP(species B)  after swapping z_C from B into A')
ax.set_ylabel('−ΔP(species A)')
ax.set_title('CUB CBM: Leakage decomposition per concept\\n'
             '(top-right = consistent with leakage)')
# Group legend (top-10 groups by count)
top_groups = swap_df['group'].value_counts().head(10).index.tolist()
ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=g) for g in top_groups],
          fontsize=7, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cub_cbm_swap_decomp.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 25: Species probe md ──────────────────────────────────────────────────────
cells.append(md("""## 7. Species-identity probe on z

Train a linear classifier `z [K-dim] → species_id (0-indexed)`.
High accuracy = concept bottleneck encodes species identity beyond concept presence."""))

# ── 26: LinearProbe ───────────────────────────────────────────────────────────
cells.append(code("""class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)


def train_classifier(X_tr, y_tr, X_te, y_te, n_classes, seed=0,
                     lr=1e-2, wd=1e-4, epochs=30, batch=512):
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
        return float((probe(X_te).argmax(1) == y_te_t).float().mean())

print('Defined: LinearProbe  train_classifier')"""))

# ── 27: Run probes ────────────────────────────────────────────────────────────
cells.append(code("""sids_te_0 = sids_te - _sid_min   # 0-indexed for probe
sids_tr_0 = sids_tr - _sid_min

# Full z -> species
acc_full = train_classifier(z_tr, sids_tr_0, z_te, sids_te_0,
                             n_classes=N_SPECIES, epochs=PROBE_EPOCHS)
print(f'Species probe on z (all {N_CONCEPTS} concepts): {acc_full:.4f}')
print(f'  chance: {1/N_SPECIES:.4f}')

# Random baseline
torch.manual_seed(42)
acc_rand = train_classifier(torch.randn_like(z_tr), sids_tr_0,
                             torch.randn_like(z_te), sids_te_0,
                             n_classes=N_SPECIES, epochs=PROBE_EPOCHS)
print(f'Species probe on random baseline: {acc_rand:.4f}')

# avgpool backbone baseline
acc_avgpool = train_classifier(avg_tr, sids_tr_0, avg_te, sids_te_0,
                                n_classes=N_SPECIES, epochs=PROBE_EPOCHS)
print(f'Species probe on avgpool (backbone): {acc_avgpool:.4f}')"""))

# ── 28: Per-concept probe ─────────────────────────────────────────────────────
cells.append(code("""# Per-concept probe: z[:, c] (single scalar) -> species_id
per_concept_acc = {}
for c_idx, c_name in enumerate(CONCEPT_NAMES):
    acc = train_classifier(
        z_tr[:, c_idx:c_idx+1], sids_tr_0,
        z_te[:, c_idx:c_idx+1], sids_te_0,
        n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
    )
    per_concept_acc[c_name] = acc

probe_acc_df = pd.DataFrame([
    {'concept': c, 'species_acc': a, 'group': ATTR_TO_GROUP[c]}
    for c, a in per_concept_acc.items()
]).sort_values('species_acc', ascending=False)
print(f'Per-concept species decodability (chance={1/N_SPECIES:.3f}):')
display(probe_acc_df.head(20))"""))

# ── 29: Species probe plots ───────────────────────────────────────────────────
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(probe_acc_df)*0.2)))

ax = axes[0]
colors_p = [CONCEPT_TO_COLOR[c] for c in probe_acc_df['concept']]
ax.barh(probe_acc_df['concept'], probe_acc_df['species_acc'], color=colors_p, alpha=0.8)
ax.axvline(1/N_SPECIES, color='red', ls='--', alpha=0.7, label=f'chance 1/{N_SPECIES}')
ax.axvline(acc_full, color='black', ls='-', alpha=0.5, label=f'all-z ({acc_full:.3f})')
ax.set_xlabel('Linear probe accuracy: z_C -> species_id')
ax.set_title('Species-identity decodability per concept activation')
ax.tick_params(axis='y', labelsize=6)
ax.grid(True, axis='x', alpha=0.3)

ax = axes[1]
labels = ['Random\\n(baseline)', f'CBM z\\n(all {N_CONCEPTS})', 'avgpool\\n(backbone)']
accs   = [acc_rand, acc_full, acc_avgpool]
ax.bar(labels, accs, color=['lightgray','steelblue','darkorange'], alpha=0.85)
ax.axhline(1/N_SPECIES, color='red', ls='--', alpha=0.7, label=f'chance 1/{N_SPECIES}')
for i, a in enumerate(accs):
    ax.text(i, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
ax.set_ylabel('Species prediction accuracy')
ax.set_title('How much species identity is in each representation?')
ax.legend(); ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, min(1.0, max(accs) * 1.2))

plt.suptitle('CUB CBM: Species-identity decodability from concept bottleneck', y=1.01)
plt.tight_layout()
plt.savefig('cub_cbm_species_probe.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 30: Binary disc md ────────────────────────────────────────────────────────
cells.append(md("""## 8. Per-pair binary discriminability in z_C

For each all-positive pair (A, B, concept C): train `z_C [scalar] → {A, B}`.
Accuracy > 0.5 = z_C distinguishes the species beyond whether the concept is present."""))

# ── 31: disc function ─────────────────────────────────────────────────────────
cells.append(code("""def binary_species_disc(sid_A, sid_B, concept_idx):
    \"\"\"Binary probe z_C -> {0=A,1=B}. Returns test accuracy. NaN if not enough data.\"\"\"
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

# ── 32: Run disc ──────────────────────────────────────────────────────────────
cells.append(code("""disc_rows = []
disc_count = 0
rng_disc = np.random.default_rng(99)

for c_idx, c_name in enumerate(CONCEPT_NAMES):
    pos_sids = np.where(cc_matrix[:, c_idx] == 1)[0] + _sid_min
    if len(pos_sids) < 2:
        continue
    all_pairs = list(combinations(pos_sids, 2))
    if len(all_pairs) > MAX_DISC_PAIRS // len(CONCEPT_NAMES):
        cap = max(1, MAX_DISC_PAIRS // len(CONCEPT_NAMES))
        idx = rng_disc.choice(len(all_pairs), cap, replace=False)
        all_pairs = [all_pairs[i] for i in idx]

    for sid_A, sid_B in all_pairs:
        acc = binary_species_disc(sid_A, sid_B, c_idx)
        if not np.isnan(acc):
            disc_rows.append({
                'sid_A': int(sid_A), 'sid_B': int(sid_B),
                'concept': c_name, 'concept_idx': c_idx,
                'group': ATTR_TO_GROUP[c_name],
                'disc_acc': float(acc),
            })
        disc_count += 1

disc_df = pd.DataFrame(disc_rows)
disc_df['disc_above_chance'] = disc_df['disc_acc'] > 0.5
print(f'Binary discriminability: {len(disc_df)} pair-concept combos')
print(f'  mean={disc_df["disc_acc"].mean():.4f}  frac>0.5={disc_df["disc_above_chance"].mean():.3f}')

disc_concept = (
    disc_df.groupby(['concept','group'])[['disc_acc','disc_above_chance']]
    .mean().reset_index().sort_values('disc_acc', ascending=False)
)
display(disc_concept.head(20))"""))

# ── 33: Disc plots ────────────────────────────────────────────────────────────
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(disc_concept)*0.2)))

ax = axes[0]
colors_d = [CONCEPT_TO_COLOR[c] for c in disc_concept['concept']]
ax.barh(disc_concept['concept'], disc_concept['disc_acc'], color=colors_d, alpha=0.8)
ax.axvline(0.5, color='red', ls='--', alpha=0.7, label='chance (0.5)')
ax.set_xlabel('Mean binary accuracy: z_C -> {species A, species B}')
ax.set_title('Per-pair species discriminability in z_C')
ax.tick_params(axis='y', labelsize=6)
ax.grid(True, axis='x', alpha=0.3)

ax = axes[1]
group_disc = (disc_df.groupby('group')['disc_acc']
              .agg(['mean','sem']).reset_index()
              .sort_values('mean', ascending=False))
ax.barh(group_disc['group'], group_disc['mean'],
        xerr=group_disc['sem'],
        color=[GROUP_COLORS[g] for g in group_disc['group']],
        alpha=0.8, capsize=4)
ax.axvline(0.5, color='red', ls='--', alpha=0.7, label='chance 0.5')
ax.set_xlabel('Mean discriminability (± SEM)')
ax.set_title('Discriminability by attribute group')
ax.legend(); ax.grid(True, axis='x', alpha=0.3)

plt.suptitle('CUB CBM: Species discriminability in z_C (per-concept scalar)', y=1.01)
plt.tight_layout()
plt.savefig('cub_cbm_binary_disc.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 34: Cross-measure md ──────────────────────────────────────────────────────
cells.append(md("""## 9. Three-level evidence: cross-measure correlation

| Analysis | Level | Measure |
|---|---|---|
| Counterfactual swap | Causal | z_C shift species prediction |
| Species-identity probe | Representational | z_C scalar predicts species |
| Binary discriminability | Mechanistic | z_C separates two species |"""))

# ── 35: Correlation ───────────────────────────────────────────────────────────
cells.append(code("""merged = (
    concept_agg[['concept','group','leakage_mean']]
    .merge(disc_concept[['concept','disc_acc']], on='concept', how='inner')
    .merge(probe_acc_df[['concept','species_acc']], on='concept', how='inner')
)

CUB_SPECIES_CSV = Path('cub_cbm_species.csv')
if CUB_SPECIES_CSV.exists():
    sp_csv = pd.read_csv(CUB_SPECIES_CSV)
    if {'attr','recall'}.issubset(sp_csv.columns):
        recall_gap = (sp_csv.groupby('attr')['recall']
                      .apply(lambda x: x.max() - x.min())
                      .reset_index(name='recall_range')
                      .rename(columns={'attr':'concept'}))
        merged = merged.merge(recall_gap, on='concept', how='left')
        print('Merged recall_range from cub_cbm_species.csv')
    else:
        merged['recall_range'] = np.nan
else:
    merged['recall_range'] = np.nan
    print('[info] cub_cbm_species.csv not found — run recall.ipynb first')

has_recall = merged['recall_range'].notna().any()
ncols  = 3 if has_recall else 2
fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4.5))

pairs_to_plot = [
    ('leakage_mean', 'disc_acc',    'Leakage vs Discriminability'),
    ('leakage_mean', 'species_acc', 'Leakage vs Species probe acc'),
]
if has_recall:
    pairs_to_plot.append(('recall_range', 'leakage_mean', 'Recall range vs Leakage'))

for ax, (x_col, y_col, title) in zip(axes, pairs_to_plot):
    sub = merged[[x_col, y_col, 'concept', 'group']].dropna()
    ax.scatter(sub[x_col], sub[y_col],
               c=[CONCEPT_TO_COLOR[c] for c in sub['concept']],
               s=40, alpha=0.7, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row['concept'].split('::')[-1],
                    (row[x_col], row[y_col]), fontsize=5, alpha=0.5)
    if len(sub) > 2:
        r, p = stats.pearsonr(sub[x_col], sub[y_col])
        m, b_coef = np.polyfit(sub[x_col], sub[y_col], 1)
        xs = np.array([sub[x_col].min(), sub[x_col].max()])
        ax.plot(xs, m*xs+b_coef, 'k--', alpha=0.5, label=f'r={r:.2f}, p={p:.3f}')
        ax.legend(fontsize=8)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title); ax.grid(True, alpha=0.3)

plt.suptitle('CUB CBM: Cross-measure correlation (three levels of evidence)', y=1.02)
plt.tight_layout()
plt.savefig('cub_cbm_evidence_correlation.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 36: Deeper diagnostics md ─────────────────────────────────────────────────
cells.append(md("""## 10. Deeper diagnostics

Addressing three questions:
1. **Does the mean mask species-specific leakage?** — histogram of per-pair leakage_sym
2. **Are only some species leaky?** — per-species mean leakage ranked
3. **Why do recall gaps exist?** — between/within-species variance of z_C for GT-positive images
4. **Do discriminability and leakage agree at the pair level?** — per-pair scatter"""))

# ── 37: Leakage distribution + per-species ────────────────────────────────────
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of per-pair leakage_sym
ax = axes[0]
vals = swap_df['leakage_sym'].values
frac_pos = (vals > 0).mean()
ax.hist(vals, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
ax.axvline(0,            color='black',  lw=1.5, ls='--', label='0')
ax.axvline(vals.mean(),  color='crimson', lw=1.5, ls='-',
           label=f'mean = {vals.mean():.4f}')
ax.text(0.97, 0.93,
        f'frac > 0: {frac_pos:.1%}\\n({(vals>0).sum()} / {len(vals)} pairs)',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.set_xlabel('leakage_sym per (A, B, concept) triple')
ax.set_ylabel('Count')
ax.set_title('Per-pair leakage distribution\\n'
             'If backwash is species-specific: expect right tail, not all-negative')
ax.legend(); ax.grid(True, alpha=0.3)

# Right: per-species mean leakage (scatter, ranked)
ax = axes[1]
sp_leak = pd.concat([
    swap_df[['sid_A','leakage_sym']].rename(columns={'sid_A':'sid'}),
    swap_df[['sid_B','leakage_sym']].rename(columns={'sid_B':'sid'}),
]).groupby('sid')['leakage_sym'].mean().reset_index(name='mean_leakage') \\
  .sort_values('mean_leakage', ascending=False).reset_index(drop=True)

sp_colors = ['crimson' if v > 0 else 'steelblue' for v in sp_leak['mean_leakage']]
ax.scatter(range(len(sp_leak)), sp_leak['mean_leakage'], c=sp_colors, s=30, zorder=3)
ax.axhline(0, color='black', lw=1, ls='--')
for rank, row in sp_leak.head(5).iterrows():
    ax.annotate(spname(int(row['sid'])).split()[-1],
                (rank, row['mean_leakage']),
                textcoords='offset points', xytext=(3,4), fontsize=6)
for rank, row in sp_leak.tail(5).iterrows():
    ax.annotate(spname(int(row['sid'])).split()[-1],
                (rank, row['mean_leakage']),
                textcoords='offset points', xytext=(3,-9), fontsize=6)
ax.set_xlabel('Species rank (highest → lowest mean leakage_sym)')
ax.set_ylabel('Mean leakage_sym')
ax.set_title('Per-species mean leakage (averaged over all concepts & pairs)\\n'
             'red = net positive leaker')
ax.grid(True, alpha=0.3)

plt.suptitle('CUB CBM: Is leakage heterogeneous across species?', y=1.02)
plt.tight_layout()
plt.savefig('cub_cbm_leakage_dist.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 38: Variance ratio ────────────────────────────────────────────────────────
cells.append(code("""# Between/within-species variance of z_C for GT-positive images
var_rows = []
for c_idx, c_name in enumerate(CONCEPT_NAMES):
    pos_sids = np.where(cc_matrix[:, c_idx] == 1)[0] + _sid_min
    sv = [z_te[sids_te == sid, c_idx].numpy()
          for sid in pos_sids if (sids_te == sid).sum() >= 1]
    if len(sv) < 2: continue
    sp_means = np.array([v.mean() for v in sv])
    sp_vars  = np.array([v.var()  for v in sv])
    var_rows.append({
        'concept': c_name, 'group': ATTR_TO_GROUP[c_name],
        'between': float(np.var(sp_means)),
        'within':  float(np.mean(sp_vars)),
        'ratio':   float(np.var(sp_means) / (np.mean(sp_vars) + 1e-8)),
    })
var_df = pd.DataFrame(var_rows).sort_values('ratio', ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(var_df)*0.15)))

# Left: variance ratio bar
ax = axes[0]
colors_vr = [CONCEPT_TO_COLOR[c] for c in var_df['concept']]
ax.barh(var_df['concept'], var_df['ratio'], color=colors_vr, alpha=0.8)
ax.axvline(1, color='red', ls='--', lw=1.2, label='ratio = 1')
ax.set_xlabel('Between-species σ² / within-species σ²\\n(GT-positive images only)')
ax.set_title('Species-specificity of z_C magnitude\\n'
             'ratio > 1 → z_C varies by species even when GT-present')
ax.tick_params(axis='y', labelsize=5)
ax.grid(True, axis='x', alpha=0.3)

# Middle & Right: z_C violin for top-2 concepts by variance ratio
top2 = var_df.head(2)['concept'].tolist()
for ax, c_name in zip(axes[1:], top2):
    c_idx    = CONCEPT_TO_IDX[c_name]
    part     = ATTR_TO_GROUP[c_name]
    pos_sids = np.where(cc_matrix[:, c_idx] == 1)[0] + _sid_min
    groups   = [(sid, z_te[sids_te == sid, c_idx].numpy())
                for sid in sorted(pos_sids) if (sids_te == sid).sum() >= 1]
    if not groups: continue
    labels_v = [spname(sid).split()[-1][:8] for sid, _ in groups]
    data_v   = [vals for _, vals in groups]
    vp = ax.violinplot(data_v, positions=range(len(data_v)),
                       showmedians=True, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(GROUP_COLORS[part]); body.set_alpha(0.55)
    vp['cmedians'].set_color('black')
    rng2 = np.random.default_rng(0)
    for i, vals in enumerate(data_v):
        jitter = rng2.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, s=8, color=GROUP_COLORS[part], alpha=0.5, zorder=3)
    ratio_val = var_df.loc[var_df.concept == c_name, 'ratio'].iloc[0]
    ax.set_xticks(range(len(labels_v)))
    ax.set_xticklabels(labels_v, rotation=60, fontsize=6)
    ax.set_ylabel('z_C value (raw logit)')
    ax.set_title(f'{c_name.split("::")[-1]}  (var ratio={ratio_val:.1f})\\n'
                 f'z_C per positive species')
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('CUB CBM: Does z_C magnitude vary by species when concept is GT-present?\\n'
             '(this directly explains recall gaps)', y=1.02)
plt.tight_layout()
plt.savefig('cub_cbm_zc_variance.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 39: Per-pair disc vs leakage ─────────────────────────────────────────────
cells.append(code("""pair_merged = swap_df[['sid_A','sid_B','concept','group','leakage_sym']].merge(
    disc_df[['sid_A','sid_B','concept','disc_acc']],
    on=['sid_A','sid_B','concept'], how='inner'
)

fig, ax = plt.subplots(figsize=(7, 5))
for group, grp in pair_merged.groupby('group'):
    ax.scatter(grp['disc_acc'], grp['leakage_sym'],
               color=GROUP_COLORS[group], s=8, alpha=0.3, label=group)
ax.axvline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
ax.axhline(0,   color='gray', ls='--', lw=1, alpha=0.7)
if len(pair_merged) > 2:
    r, p = stats.pearsonr(pair_merged['disc_acc'], pair_merged['leakage_sym'])
    m, b_coef = np.polyfit(pair_merged['disc_acc'], pair_merged['leakage_sym'], 1)
    xs = np.array([pair_merged['disc_acc'].min(), pair_merged['disc_acc'].max()])
    ax.plot(xs, m*xs+b_coef, 'k-', lw=1.5, alpha=0.8, label=f'r={r:.2f}, p={p:.3f}')
ax.set_xlabel('Binary discriminability (z_C → {A,B} accuracy)\\nper (species A, species B, concept) triple')
ax.set_ylabel('leakage_sym\\n(>0 = swap raised P(B))')
ax.set_title('CUB CBM: Per-pair discriminability vs leakage\\n'
             'If both measure the same thing: expect positive correlation')
# top-8 groups legend
top8 = swap_df['group'].value_counts().head(8).index
ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=g) for g in top8],
          fontsize=7, markerscale=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cub_cbm_pairwise_disc_vs_leakage.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ── 40: Summary md ────────────────────────────────────────────────────────────
cells.append(md("""## 11. Summary

### What we measured (CUB CBM = MCBM γ=0)

| Measure | Value | Interpretation |
|---|---|---|
| Species probe on z (all K) | `acc_full` | Species info in concept bottleneck |
| Species probe on avgpool | `acc_avgpool` | Upstream ceiling |
| Mean leakage_sym (swap) | from swap_df | Causal: z_C carries species signal |
| Mean binary discriminability | from disc_df | z_C scalar separates species |

### Direct comparison with FunnyBirds CBM

| Aspect | FunnyBirds CBM | CUB CBM (MCBM γ=0) |
|---|---|---|
| Species | 50 | 200 |
| Concepts | 26 exact binary | ~100 attribute majority vote |
| z computation | sigmoid(W_c @ avgpool) | W_c @ avgpool (raw logits) |
| CC matrix | exact GT | training majority vote |
| Recall gap source | discriminability of z_C | same mechanism, noisier labels |

### Key questions
- Does leakage_sym distribution have a right tail (species-specific backwash)?
- Is the variance ratio > 1 for certain attribute groups (concept-specific backwash)?
- Does the negative leakage/discriminability correlation replicate from FunnyBirds?

### Next: MCBM γ sweep
Repeat with `checkpoints_mcbm/mcbm_gamma{X}.pth` — the IB penalty should reduce
`leakage_sym` and `disc_acc` while preserving concept accuracy."""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
    },
    "cells": cells,
}

out = Path("notebooks/cub_cbm_counterfactual.ipynb")
out.parent.mkdir(exist_ok=True)
with out.open("w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {out}  ({len(cells)} cells)")
