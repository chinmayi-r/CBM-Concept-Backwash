#!/usr/bin/env python3
"""Generate notebooks/funnybirds_mcbm_counterfactual.ipynb."""
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
# FunnyBirds MCBM — Counterfactual Concept Swap — γ sweep

Parallel to `funnybirds_cbm_counterfactual.ipynb` but extended to **MCBM** (γ sweep).

**Central hypothesis:** Higher γ (stronger IB penalty) → z_C retains less species-identity
signal → lower leakage score across all three analyses.

Three analyses, each run for every γ:

1. **Counterfactual concept swap** — replace z_C in species A with z_C from species B;
   measure the induced shift in P(species B).  `leakage_sym > 0` = z_C carries species signal.
2. **Species-identity probe** — linear `z [26-dim] → species_id` accuracy.
   High accuracy = the bottleneck encodes species, not just concepts.
3. **Per-concept binary discriminability** — `z_C [scalar] → {A, B}`.
   Above-chance = that single concept activation distinguishes two species.

γ=0.0 recovers the standard CBM and provides a direct sanity-check against
`funnybirds_cbm_counterfactual.ipynb`."""))

# ── Cell 1: imports ───────────────────────────────────────────────────────────
cells.append(code("""\
import random as _random
import sys
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

# ── Cell 2: section header ────────────────────────────────────────────────────
cells.append(md("## 0. Configuration"))

# ── Cell 3: config ────────────────────────────────────────────────────────────
cells.append(code("""\
GAMMAS       = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
N_SPECIES    = 50
N_CONCEPTS   = 26
PROBE_EPOCHS = 30   # epochs for all linear probes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[config] device: {device}')
print(f'[config] gammas: {GAMMAS}')"""))

# ── Cell 4: paths header ──────────────────────────────────────────────────────
cells.append(md("""\
## 1. Paths

- `FB` — FunnyBirds dataset root
- `MCBM_FEATS[γ]` — avgpool features extracted from the MCBM backbone trained at strength γ
- `MCBM_CKPTS[γ]` — MCBM checkpoint for γ (concept_head + label_head weights)

Gammas where either features or checkpoint are missing are silently excluded."""))

# ── Cell 5: paths ─────────────────────────────────────────────────────────────
cells.append(code("""\
ROOT = Path('/scratch/network/cr7998/cv_emergence_project')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FB         = ROOT / 'data' / 'FunnyBirds'
MCBM_FEATS = {g: ROOT / 'features' / f'resnet50_mcbm_funnybirds_gamma{g}'
              for g in GAMMAS}
MCBM_CKPTS = {g: ROOT / 'checkpoints_funnybirds' / f'mcbm_funnybirds_gamma{g}.pth'
              for g in GAMMAS}


def _feats_ready(p: Path) -> bool:
    return (p / 'labels_train.pt').exists() and (p / 'labels_test.pt').exists()


_missing = [g for g in GAMMAS
            if not _feats_ready(MCBM_FEATS[g]) or not MCBM_CKPTS[g].exists()]
if _missing:
    print(f'[warn] gamma={_missing}: features or checkpoint missing — excluding.')
    GAMMAS = [g for g in GAMMAS
              if _feats_ready(MCBM_FEATS[g]) and MCBM_CKPTS[g].exists()]
if not GAMMAS:
    raise RuntimeError('No MCBM checkpoints available. Run MCBM training first.')

assert FB.exists(), f'Missing FunnyBirds folder: {FB}'
assert (FB / 'dataset_train.json').exists(), f'Missing dataset_train.json'
print(f'[config] available gammas: {GAMMAS}')"""))

# ── Cell 6: metadata header ───────────────────────────────────────────────────
cells.append(md("## 2. Metadata, concept names, class-concept matrix"))

# ── Cell 7: load dataset stuff ────────────────────────────────────────────────
cells.append(code("""\
from datasets.funnybirds_dataset import (
    FunnyBirdsDataset, concept_names as _cnames,
)

CONCEPT_NAMES  = _cnames()   # list of 26 strings, e.g. ['beak_0', 'beak_1', ...]
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

# Exact class-concept matrix: cc_matrix[s, c] = 1 iff species s truly has concept c
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

# ── Cell 8: per-image metadata ────────────────────────────────────────────────
cells.append(code("""\
meta    = pd.read_csv(FB / 'metadata' / 'images.csv')
id2name = dict(pd.read_csv(FB / 'metadata' / 'classes.csv')
               .pipe(lambda d: zip(d['class_id'], d['class_name'])))
meta['species_name'] = meta['class_id'].map(id2name)
meta_test = meta[meta['is_train'] == 0].copy()

# image_id → species_id for split-order alignment
imgid_to_sid = dict(zip(meta['image_id'], meta['class_id']))

print(f'Test images: {len(meta_test)}  ({meta_test["class_id"].nunique()} species)')"""))

# ── Cell 9: weights header ────────────────────────────────────────────────────
cells.append(md("""\
## 3. Load MCBM weights and compute concept activations

For each γ we load:
- **avgpool features** — `MCBM_FEATS[γ]/avgpool_{split}.pt`  (backbone output, γ-specific)
- **concept_head** — `W_c [K, 2048]`, `b_c [K]`  (from checkpoint)
- **label_head**   — `W_y [50, K]`, `b_y [50]`   (from checkpoint)

Then: `z_γ = sigmoid(avgpool_γ @ W_c.T + b_c)` — the deterministic concept activation
(mean of the IB distribution; γ only affects training, not the inference forward pass).

The DataLoader ordering is verified to be identical across all γ so `imgid_to_sid`
can be applied once."""))

# ── Cell 10: loader helpers ───────────────────────────────────────────────────
cells.append(code("""\
def _load(p: Path):
    try:
        return torch.load(p, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(p, map_location='cpu')


def load_mcbm_heads(ckpt_path: Path):
    \"\"\"
    Load concept_head and label_head from an MCBM checkpoint.
    Tries multiple key prefixes: concept_head, mean_head, bottleneck.
    Returns: W_c [K, D], b_c [K], W_y [n_sp, K], b_y [n_sp].
    \"\"\"
    ckpt = _load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)

    for w_key in ['concept_head.weight', 'mean_head.weight', 'bottleneck.weight']:
        if w_key in sd:
            b_key = w_key.replace('weight', 'bias')
            W_c   = sd[w_key].float()
            b_c   = sd[b_key].float()
            break
    else:
        heads = [k for k in sd if 'weight' in k][:10]
        raise KeyError(f'concept/mean/bottleneck head not found. Keys: {heads}')

    W_y = sd['label_head.weight'].float()
    b_y = sd['label_head.bias'].float()
    return W_c, b_c, W_y, b_y


def load_avgpool_split(feat_dir: Path, split: str) -> torch.Tensor:
    p = feat_dir / f'avgpool_{split}.pt'
    assert p.exists(), f'Missing avgpool features: {p}'
    X = _load(p)
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    return X.float()


def load_image_ids_split(feat_dir: Path, split: str) -> np.ndarray:
    p = feat_dir / f'labels_{split}.pt'
    assert p.exists(), f'Missing labels: {p}'
    t = _load(p)
    ids = t['image_ids'] if isinstance(t, dict) else t
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    return np.asarray(ids).reshape(-1).astype(int)


@torch.no_grad()
def compute_z(avgpool: torch.Tensor, W_c: torch.Tensor, b_c: torch.Tensor) -> torch.Tensor:
    \"\"\"z = sigmoid(avgpool @ W_c.T + b_c)  [N, K]\"\"\"
    return torch.sigmoid(avgpool @ W_c.T + b_c)


@torch.no_grad()
def compute_logits(z: torch.Tensor, W_y: torch.Tensor, b_y: torch.Tensor) -> torch.Tensor:
    \"\"\"logits = z @ W_y.T + b_y  [N, n_species]\"\"\"
    return z @ W_y.T + b_y


print('Defined: _load  load_mcbm_heads  load_avgpool_split  load_image_ids_split')
print('         compute_z  compute_logits')"""))

# ── Cell 11: load all gamma data ──────────────────────────────────────────────
cells.append(code("""\
# Verify all gammas share the same DataLoader order, then map to species IDs
_ref_g  = GAMMAS[0]
ids_te  = load_image_ids_split(MCBM_FEATS[_ref_g], 'test')
ids_tr  = load_image_ids_split(MCBM_FEATS[_ref_g], 'train')

for g in GAMMAS[1:]:
    te_g = load_image_ids_split(MCBM_FEATS[g], 'test')
    tr_g = load_image_ids_split(MCBM_FEATS[g], 'train')
    assert np.array_equal(ids_te, te_g), f'gamma={g} test order differs from gamma={_ref_g}'
    assert np.array_equal(ids_tr, tr_g), f'gamma={g} train order differs from gamma={_ref_g}'
print(f'Split order consistent across all {len(GAMMAS)} gammas.')
print(f'Test N={len(ids_te)}  Train N={len(ids_tr)}')

sids_te = np.array([imgid_to_sid[int(i)] for i in ids_te])
sids_tr = np.array([imgid_to_sid[int(i)] for i in ids_tr])

# Per-gamma: load backbone avgpool + compute z from checkpoint weights
gamma_data = {}
for g in GAMMAS:
    avg_te_g = load_avgpool_split(MCBM_FEATS[g], 'test')
    avg_tr_g = load_avgpool_split(MCBM_FEATS[g], 'train')
    W_c, b_c, W_y, b_y = load_mcbm_heads(MCBM_CKPTS[g])
    z_te_g   = compute_z(avg_te_g, W_c, b_c)
    z_tr_g   = compute_z(avg_tr_g, W_c, b_c)
    logits_g = compute_logits(z_te_g, W_y, b_y)
    sp_acc   = float((logits_g.argmax(1).numpy() == sids_te).mean())
    gamma_data[g] = {
        'W_c': W_c, 'b_c': b_c, 'W_y': W_y, 'b_y': b_y,
        'z_te': z_te_g, 'z_tr': z_tr_g,
        'test_sp_acc': sp_acc,
    }
    print(f'  gamma={g:4.2f}: z_te {tuple(z_te_g.shape)}  species_acc={sp_acc:.4f}')"""))

# ── Cell 12: sanity checks ────────────────────────────────────────────────────
cells.append(code("""\
# Sanity: concept accuracy (z > 0.5 vs GT class-concept matrix)
print('Concept accuracy per gamma (threshold=0.5 vs GT cc_matrix):')
print(f'  {\"gamma\":>6}  concept_acc  species_acc')
for g in GAMMAS:
    z_te_g       = gamma_data[g]['z_te']
    gt_concepts  = cc_matrix[sids_te]                       # [N_te, 26]
    pred         = (z_te_g.numpy() > 0.5).astype(int)
    concept_acc  = float((pred == gt_concepts).mean())
    print(f'  {g:>6.2f}  {concept_acc:.4f}       {gamma_data[g][\"test_sp_acc\"]:.4f}')"""))

# ── Cell 13: swap header ──────────────────────────────────────────────────────
cells.append(md("""\
## 4. Counterfactual concept swap

For an all-positive pair (species A, species B, concept C):

```
z_swapped = z_A  but  z_swapped[:, C] ← z_B[:, C]
logits_swapped = z_swapped @ W_y.T + b_y
shift_B = mean(softmax(logits_swapped)[:, B]) − mean(softmax(logits_A)[:, B])
```

`leakage_sym = 0.5 * (leakage_fwd + leakage_bwd)` averages both swap directions.
`leakage_sym > 0` means z_C carries species-identity signal beyond the concept itself.

The **IB hypothesis**: as γ increases, z_C is compressed toward a minimal sufficient
statistic for concept C, so species-identity signal is squeezed out → leakage_sym ↓."""))

# ── Cell 14: swap functions ───────────────────────────────────────────────────
cells.append(code("""\
@torch.no_grad()
def concept_swap(
    z_A: torch.Tensor,   # [N_A, K]  recipient
    z_B: torch.Tensor,   # [N_B, K]  donor
    concept_idx: int,
    W_y: torch.Tensor,   # [n_sp, K]
    b_y: torch.Tensor,   # [n_sp]
):
    \"\"\"
    For every A-image, average swap over all N_B donor B-images.
    Returns p_orig [N_A, n_sp] and p_swap [N_A, n_sp].
    \"\"\"
    N_A, K = z_A.shape
    N_B    = z_B.shape[0]

    p_orig = torch.softmax(z_A @ W_y.T + b_y, dim=-1)          # [N_A, n_sp]

    z_swap = z_A.unsqueeze(1).expand(N_A, N_B, K).clone()      # [N_A, N_B, K]
    z_swap[:, :, concept_idx] = z_B[:, concept_idx].unsqueeze(0)

    logits_swap = z_swap.view(N_A * N_B, K) @ W_y.T + b_y      # [N_A*N_B, n_sp]
    p_swap = torch.softmax(logits_swap, dim=-1).view(N_A, N_B, -1).mean(dim=1)

    return p_orig, p_swap


def pair_swap_metrics(
    sid_A: int, sid_B: int, concept_idx: int,
    z_te: torch.Tensor, sids_te: np.ndarray,
    W_y: torch.Tensor, b_y: torch.Tensor,
):
    \"\"\"
    Symmetric leakage score for triple (species A, species B, concept C).
    Returns None if either species has no test images.
    \"\"\"
    mask_A = sids_te == sid_A
    mask_B = sids_te == sid_B
    z_A    = z_te[mask_A]
    z_B    = z_te[mask_B]
    if len(z_A) == 0 or len(z_B) == 0:
        return None

    # A→B: put B's z_C into A
    p_orig_A, p_swap_A = concept_swap(z_A, z_B, concept_idx, W_y, b_y)
    shift_B_fwd = (p_swap_A[:, sid_B] - p_orig_A[:, sid_B]).mean().item()
    shift_A_fwd = (p_swap_A[:, sid_A] - p_orig_A[:, sid_A]).mean().item()

    # B→A: put A's z_C into B
    p_orig_B, p_swap_B = concept_swap(z_B, z_A, concept_idx, W_y, b_y)
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

# ── Cell 15: run swap header ──────────────────────────────────────────────────
cells.append(md("""\
## 5. Run swaps for all GT-positive pairs — per γ

For each concept C: all species that truly have C=1 (from class-concept matrix),
all pairs among them, both swap directions."""))

# ── Cell 16: run swap loop ────────────────────────────────────────────────────
cells.append(code("""\
swap_results = {}   # gamma -> DataFrame

for g in GAMMAS:
    print(f'\\nCounterfactual swap  gamma={g}')
    gd   = gamma_data[g]
    rows = []
    for c_idx, c_name in enumerate(CONCEPT_NAMES):
        positive_sids = np.where(cc_matrix[:, c_idx] == 1)[0]
        if len(positive_sids) < 2:
            continue
        for sid_A, sid_B in combinations(positive_sids, 2):
            r = pair_swap_metrics(
                sid_A, sid_B, c_idx,
                gd['z_te'], sids_te,
                gd['W_y'], gd['b_y'],
            )
            if r is not None:
                rows.append(r)
    df          = pd.DataFrame(rows)
    df['part']  = df['concept'].map(CONCEPT_TO_PART)
    df['gamma'] = float(g)
    swap_results[g] = df
    mean_l = df['leakage_sym'].mean()
    frac_p = float((df['leakage_sym'] > 0).mean())
    print(f'  gamma={g}: {len(df)} pairs  '
          f'mean_leakage={mean_l:.4f}  frac_positive={frac_p:.3f}')

for g, df in swap_results.items():
    df.to_csv(f'fb_mcbm_swap_gamma{g}.csv', index=False)
    print(f'Saved fb_mcbm_swap_gamma{g}.csv')"""))

# ── Cell 17: per-concept aggregation ─────────────────────────────────────────
cells.append(code("""\
# Per-concept leakage aggregation per gamma
concept_leakage = {}   # gamma -> concept-level agg DataFrame

for g in GAMMAS:
    df  = swap_results[g]
    agg = (
        df.groupby(['concept', 'part'], as_index=False)
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
    concept_leakage[g] = agg

print('Per-gamma mean leakage (concept-level aggregation):')
for g in GAMMAS:
    agg = concept_leakage[g]
    print(f'  gamma={g:4.2f}: mean={agg[\"leakage_mean\"].mean():.4f}  '
          f'max={agg[\"leakage_mean\"].max():.4f}')"""))

# ── Cell 18: γ-sweep header ───────────────────────────────────────────────────
cells.append(md("## 6. γ-sweep main result: leakage score vs γ"))

# ── Cell 19: main leakage vs gamma plot ───────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: mean leakage ± SEM vs gamma
ax = axes[0]
mean_lk = [swap_results[g]['leakage_sym'].mean()  for g in GAMMAS]
sem_lk  = [swap_results[g]['leakage_sym'].std() /
           np.sqrt(len(swap_results[g])) for g in GAMMAS]
ax.errorbar(GAMMAS, mean_lk, yerr=sem_lk, marker='o', color='steelblue', capsize=4, lw=2)
ax.fill_between(GAMMAS,
                [m - s for m, s in zip(mean_lk, sem_lk)],
                [m + s for m, s in zip(mean_lk, sem_lk)],
                alpha=0.15, color='steelblue')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('γ (IB penalty strength)')
ax.set_ylabel('Mean leakage score (± SEM)')
ax.set_title('Does IB reduce concept-level leakage?\\n'
             '(main hypothesis: decreases with γ)')
ax.grid(True, alpha=0.3)

# Panel B: per-concept lines coloured by part
ax = axes[1]
for c_name in CONCEPT_NAMES:
    vals = []
    for g in GAMMAS:
        agg = concept_leakage[g]
        row = agg[agg['concept'] == c_name]
        vals.append(float(row['leakage_mean'].iloc[0]) if len(row) > 0 else np.nan)
    part = CONCEPT_TO_PART.get(c_name, 'unknown')
    ax.plot(GAMMAS, vals, alpha=0.55, color=PART_COLORS.get(part, 'gray'), lw=1.2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('γ (IB penalty strength)')
ax.set_ylabel('Mean leakage score per concept')
ax.set_title('Per-concept leakage vs γ (coloured by body part)')
ax.legend(handles=[Patch(color=c, label=p) for p, c in PART_COLORS.items()], fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('FunnyBirds MCBM: Counterfactual leakage score vs γ', y=1.02)
plt.tight_layout()
plt.savefig('fb_mcbm_leakage_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_leakage_vs_gamma.png')"""))

# ── Cell 20: leakage heatmap ──────────────────────────────────────────────────
cells.append(code("""\
# Heatmap: concept (rows) × gamma (columns)
heat_data = pd.DataFrame(
    {g: concept_leakage[g].set_index('concept')['leakage_mean'] for g in GAMMAS}
)
heat_data.columns = [f'γ={g}' for g in GAMMAS]

vals = heat_data.values[~np.isnan(heat_data.values)]
vmax = max(abs(vals.max()), abs(vals.min())) if len(vals) else 0.1

fig, ax = plt.subplots(figsize=(len(GAMMAS) * 1.3 + 2, len(CONCEPT_NAMES) * 0.35 + 1))
im = ax.imshow(heat_data.values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(GAMMAS)));  ax.set_xticklabels(heat_data.columns, rotation=30, ha='right')
ax.set_yticks(range(len(heat_data))); ax.set_yticklabels(heat_data.index, fontsize=8)
ax.set_title('Counterfactual leakage per concept × γ\\n'
             '(red = positive leakage, blue = negative)')
plt.colorbar(im, ax=ax, label='mean leakage_sym')
plt.tight_layout()
plt.savefig('fb_mcbm_leakage_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_leakage_heatmap.png')"""))

# ── Cell 21: compare γ=0 vs γ_max per-concept barh ────────────────────────────
cells.append(code("""\
# Per-concept leakage: gamma=0 (CBM-equiv) vs max gamma — side-by-side barh
g_lo, g_hi = GAMMAS[0], GAMMAS[-1]
agg_lo = concept_leakage[g_lo]
agg_hi = concept_leakage[g_hi]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, agg, title in [
    (axes[0], agg_lo, f'γ={g_lo}  (CBM-equivalent)'),
    (axes[1], agg_hi, f'γ={g_hi}  (max IB)'),
]:
    colors = [PART_COLORS[p] for p in agg['part']]
    ax.barh(agg['concept'], agg['leakage_mean'], color=colors, alpha=0.8)
    ax.errorbar(
        agg['leakage_mean'], range(len(agg)),
        xerr=agg['leakage_std'].fillna(0),
        fmt='none', color='black', alpha=0.5, capsize=3,
    )
    ax.axvline(0, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('Mean leakage score (shift_B − shift_A, symmetric)')
    ax.set_title(title)
    ax.legend(
        handles=[Patch(color=c, label=p) for p, c in PART_COLORS.items()],
        fontsize=7, loc='lower right',
    )
    ax.grid(True, axis='x', alpha=0.3)

plt.suptitle(f'FunnyBirds MCBM: Per-concept leakage — γ={g_lo} vs γ={g_hi}', y=1.02)
plt.tight_layout()
plt.savefig('fb_mcbm_leakage_byconc_compare.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_leakage_byconc_compare.png')"""))

# ── Cell 22: swap decomposition scatter ──────────────────────────────────────
cells.append(code("""\
# Decomposition scatter: ΔP(B) vs −ΔP(A) for each concept — compare gamma=0 vs max
fig, axes = plt.subplots(1, len([g_lo, g_hi]), figsize=(12, 5))

for ax, g in zip(axes, [g_lo, g_hi]):
    agg = concept_leakage[g]
    for _, row in agg.iterrows():
        color = PART_COLORS.get(row['part'], 'gray')
        ax.scatter(row['shift_B_mean'], -row['shift_A_mean'],
                   color=color, s=60, alpha=0.8, zorder=3)
        ax.annotate(row['concept'], (row['shift_B_mean'], -row['shift_A_mean']),
                    fontsize=6, alpha=0.7)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('ΔP(species B)  after A→B swap  (>0 if leakage)')
    ax.set_ylabel('−ΔP(species A)  (>0 if leakage)')
    ax.set_title(f'Leakage decomposition  γ={g}\\n'
                 '(top-right = both effects consistent with leakage)')
    ax.legend(handles=[Patch(color=c, label=p) for p, c in PART_COLORS.items()], fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('FunnyBirds MCBM: Swap decomposition per concept', y=1.02)
plt.tight_layout()
plt.savefig('fb_mcbm_swap_decomp_compare.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_swap_decomp_compare.png')"""))

# ── Cell 23: species probe header ────────────────────────────────────────────
cells.append(md("""\
## 7. Species-identity probe on z

Train a linear classifier `z [26-dim] → species_id` on the CBM/MCBM bottleneck activations.
High accuracy = species identity is encoded in z, not just the concepts.

Run for every γ and compare to a random-Gaussian baseline (same shape)."""))

# ── Cell 24: LinearProbe + train_classifier ───────────────────────────────────
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
    torch.manual_seed(seed); np.random.seed(seed); _random.seed(seed)
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

# ── Cell 25: species probe per gamma ─────────────────────────────────────────
cells.append(code("""\
probe_results = {}   # gamma -> {'acc_full': float, 'acc_rand': float}

for g in GAMMAS:
    z_te_g = gamma_data[g]['z_te']
    z_tr_g = gamma_data[g]['z_tr']

    acc_full = train_classifier(
        z_tr_g, sids_tr, z_te_g, sids_te,
        n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
    )
    torch.manual_seed(42)
    acc_rand = train_classifier(
        torch.randn_like(z_tr_g), sids_tr,
        torch.randn_like(z_te_g), sids_te,
        n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
    )
    probe_results[g] = {'acc_full': acc_full, 'acc_rand': acc_rand}
    print(f'  gamma={g:4.2f}: z-probe={acc_full:.4f}  '
          f'random={acc_rand:.4f}  chance={1/N_SPECIES:.4f}')"""))

# ── Cell 26: species probe vs gamma plot ──────────────────────────────────────
cells.append(code("""\
fig, ax = plt.subplots(figsize=(7, 4))

accs_full = [probe_results[g]['acc_full'] for g in GAMMAS]
accs_rand = [probe_results[g]['acc_rand'] for g in GAMMAS]

ax.plot(GAMMAS, accs_full, marker='o', color='steelblue', lw=2, label='z (all 26 concepts)')
ax.plot(GAMMAS, accs_rand, marker='s', color='gray', ls='--', lw=1.2, label='random baseline')
ax.axhline(1/N_SPECIES, color='red', ls=':', alpha=0.7, label=f'chance (1/{N_SPECIES})')
ax.set_xlabel('γ (IB penalty strength)')
ax.set_ylabel('Species probe accuracy (linear z → species_id)')
ax.set_title('FunnyBirds MCBM: Species identity encoded in z vs γ\\n'
             '(decreasing = IB suppresses species leakage through bottleneck)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fb_mcbm_species_probe_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_species_probe_vs_gamma.png')"""))

# ── Cell 27: per-concept species probe per gamma ──────────────────────────────
cells.append(code("""\
# Per-concept: z[:, c] → species_id (single scalar per image)
per_concept_probe = {}   # gamma -> {concept_name: acc}

for g in GAMMAS:
    z_te_g = gamma_data[g]['z_te']
    z_tr_g = gamma_data[g]['z_tr']
    cdict  = {}
    for c_idx, c_name in enumerate(CONCEPT_NAMES):
        acc = train_classifier(
            z_tr_g[:, c_idx:c_idx+1], sids_tr,
            z_te_g[:, c_idx:c_idx+1], sids_te,
            n_classes=N_SPECIES, epochs=PROBE_EPOCHS,
        )
        cdict[c_name] = acc
    per_concept_probe[g] = cdict
    print(f'  gamma={g:4.2f}: mean per-concept species acc='
          f'{np.mean(list(cdict.values())):.4f}')"""))

# ── Cell 28: per-concept probe plot ───────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Panel A: per-concept barh — gamma=0 vs max gamma
ax = axes[0]
y_pos  = np.arange(N_CONCEPTS)
acc_lo = [per_concept_probe[GAMMAS[0]].get(c, np.nan) for c in CONCEPT_NAMES]
acc_hi = [per_concept_probe[GAMMAS[-1]].get(c, np.nan) for c in CONCEPT_NAMES]
w = 0.35
ax.barh(y_pos - w/2, acc_lo, w, color='steelblue',   alpha=0.8, label=f'γ={GAMMAS[0]}')
ax.barh(y_pos + w/2, acc_hi, w, color='darkorange',  alpha=0.8, label=f'γ={GAMMAS[-1]}')
ax.set_yticks(y_pos); ax.set_yticklabels(CONCEPT_NAMES, fontsize=7)
ax.axvline(1/N_SPECIES, color='red', ls='--', alpha=0.7, label=f'chance 1/{N_SPECIES}')
ax.set_xlabel('Species probe acc (single concept scalar → species_id)')
ax.set_title(f'Per-concept species decodability: γ={GAMMAS[0]} vs γ={GAMMAS[-1]}')
ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

# Panel B: mean across concepts vs gamma
ax = axes[1]
means = [np.mean(list(per_concept_probe[g].values())) for g in GAMMAS]
sems  = [np.std(list(per_concept_probe[g].values())) / np.sqrt(N_CONCEPTS) for g in GAMMAS]
ax.errorbar(GAMMAS, means, yerr=sems, marker='o', capsize=4, color='steelblue')
ax.axhline(1/N_SPECIES, color='red', ls='--', alpha=0.7, label=f'chance 1/{N_SPECIES}')
ax.set_xlabel('γ (IB penalty strength)')
ax.set_ylabel('Mean per-concept species probe acc (± SEM)')
ax.set_title('Mean species decodability per concept scalar vs γ')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('FunnyBirds MCBM: Per-concept species decodability vs γ', y=1.02)
plt.tight_layout()
plt.savefig('fb_mcbm_perconcept_probe_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_perconcept_probe_vs_gamma.png')"""))

# ── Cell 29: binary disc header ───────────────────────────────────────────────
cells.append(md("""\
## 8. Per-pair binary discriminability in z_C

For each all-positive pair (A, B, concept C):
train a binary classifier `z_C [scalar] → {species A, species B}`.

Accuracy > 0.5 = z_C alone distinguishes the two species — direct mechanistic evidence
that the concept activation carries species-specific signal beyond whether the concept
is present (both species truly have C=1)."""))

# ── Cell 30: binary_species_disc ─────────────────────────────────────────────
cells.append(code("""\
def binary_species_disc(
    sid_A: int, sid_B: int, concept_idx: int,
    z_tr: torch.Tensor, sids_tr: np.ndarray,
    z_te: torch.Tensor, sids_te: np.ndarray,
    epochs: int = PROBE_EPOCHS,
) -> float:
    \"\"\"
    Train z[:, concept_idx] → {0=A, 1=B} binary probe.
    Returns test accuracy (0.5 = chance).  NaN if not enough data.
    \"\"\"
    mask_tr = np.isin(sids_tr, [sid_A, sid_B])
    mask_te = np.isin(sids_te, [sid_A, sid_B])
    if mask_tr.sum() < 4 or mask_te.sum() < 2:
        return float('nan')

    X_tr = z_tr[mask_tr, concept_idx:concept_idx+1]   # [N_tr_sub, 1]
    X_te = z_te[mask_te, concept_idx:concept_idx+1]   # [N_te_sub, 1]
    y_tr = (sids_tr[mask_tr] == sid_B).astype(int)
    y_te = (sids_te[mask_te] == sid_B).astype(int)

    return train_classifier(X_tr, y_tr, X_te, y_te, n_classes=2, epochs=epochs)


print('Defined: binary_species_disc')"""))

# ── Cell 31: run disc per gamma ───────────────────────────────────────────────
cells.append(code("""\
disc_results = {}   # gamma -> DataFrame

for g in GAMMAS:
    print(f'\\nBinary discriminability  gamma={g}')
    gd   = gamma_data[g]
    rows = []
    for c_idx, c_name in enumerate(CONCEPT_NAMES):
        positive_sids = np.where(cc_matrix[:, c_idx] == 1)[0]
        if len(positive_sids) < 2:
            continue
        for sid_A, sid_B in combinations(positive_sids, 2):
            acc = binary_species_disc(
                sid_A, sid_B, c_idx,
                gd['z_tr'], sids_tr,
                gd['z_te'], sids_te,
            )
            if not np.isnan(acc):
                rows.append({
                    'sid_A': int(sid_A), 'sid_B': int(sid_B),
                    'concept': c_name, 'concept_idx': c_idx,
                    'part': CONCEPT_TO_PART[c_name],
                    'disc_acc': float(acc),
                    'gamma': float(g),
                })
    df = pd.DataFrame(rows)
    disc_results[g] = df
    mean_d = df['disc_acc'].mean() if len(df) > 0 else float('nan')
    frac_d = float((df['disc_acc'] > 0.5).mean()) if len(df) > 0 else float('nan')
    print(f'  gamma={g}: {len(df)} pair-concepts  '
          f'mean_disc={mean_d:.4f}  frac>0.5={frac_d:.3f}')"""))

# ── Cell 32: disc plots ───────────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: mean discriminability ± SEM vs gamma
ax = axes[0]
means_d = [disc_results[g]['disc_acc'].mean()  for g in GAMMAS]
sems_d  = [disc_results[g]['disc_acc'].std() /
           np.sqrt(len(disc_results[g]))        for g in GAMMAS]
ax.errorbar(GAMMAS, means_d, yerr=sems_d, marker='o', capsize=4, color='steelblue', lw=2)
ax.fill_between(GAMMAS,
                [m - s for m, s in zip(means_d, sems_d)],
                [m + s for m, s in zip(means_d, sems_d)],
                alpha=0.15, color='steelblue')
ax.axhline(0.5, color='red', ls='--', alpha=0.7, label='chance 0.5')
ax.set_xlabel('γ (IB penalty strength)')
ax.set_ylabel('Mean binary discriminability (± SEM)')
ax.set_title('Species discriminability in z_C vs γ\\n'
             '(hypothesis: decreases as IB compresses species signal out of z_C)')
ax.legend(); ax.grid(True, alpha=0.3)

# Panel B: per-part discriminability vs gamma
ax = axes[1]
for part in ['beak', 'wing', 'tail', 'foot', 'eye']:
    vals = []
    for g in GAMMAS:
        sub = disc_results[g][disc_results[g]['part'] == part]['disc_acc'].dropna()
        vals.append(float(sub.mean()) if len(sub) > 0 else float('nan'))
    ax.plot(GAMMAS, vals, marker='o', label=part, color=PART_COLORS.get(part), alpha=0.8)
ax.axhline(0.5, color='red', ls='--', alpha=0.7, label='chance 0.5')
ax.set_xlabel('γ (IB penalty strength)')
ax.set_ylabel('Mean binary discriminability by part')
ax.set_title('Discriminability by body part vs γ')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('FunnyBirds MCBM: Binary species discriminability in z_C vs γ', y=1.02)
plt.tight_layout()
plt.savefig('fb_mcbm_discrim_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_discrim_vs_gamma.png')"""))

# ── Cell 33: summary header ───────────────────────────────────────────────────
cells.append(md("""\
## 9. Three-level evidence summary — cross-γ

| Analysis | Level | Measure | Prediction |
|---|---|---|---|
| Counterfactual swap | Causal | `leakage_sym` | ↓ with γ |
| Species-identity probe | Global | `acc_full` (z → species) | ↓ with γ |
| Binary discriminability | Per-concept | `disc_acc` (z_C → pair) | ↓ with γ |"""))

# ── Cell 34: summary table ────────────────────────────────────────────────────
cells.append(code("""\
summary_rows = []
for g in GAMMAS:
    mean_leak = float(swap_results[g]['leakage_sym'].mean())
    frac_pos  = float((swap_results[g]['leakage_sym'] > 0).mean())
    acc_z     = probe_results[g]['acc_full']
    mean_disc = float(disc_results[g]['disc_acc'].mean()) if len(disc_results[g]) > 0 else float('nan')
    frac_disc = float((disc_results[g]['disc_acc'] > 0.5).mean()) if len(disc_results[g]) > 0 else float('nan')
    summary_rows.append({
        'gamma': g,
        'mean_leakage_sym':  round(mean_leak, 5),
        'frac_leakage_pos':  round(frac_pos,  3),
        'species_probe_acc': round(acc_z,      4),
        'mean_disc_acc':     round(mean_disc,  4),
        'frac_disc_above_chance': round(frac_disc, 3),
    })

summary_df = pd.DataFrame(summary_rows)
print('Cross-γ summary — three levels of evidence:')
display(summary_df)
summary_df.to_csv('fb_mcbm_counterfactual_summary.csv', index=False)
print('Saved fb_mcbm_counterfactual_summary.csv')"""))

# ── Cell 35: final three-panel figure ─────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel 1: leakage
ax = axes[0]
ax.errorbar(summary_df['gamma'], summary_df['mean_leakage_sym'],
            marker='o', capsize=4, color='steelblue', lw=2)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('γ'); ax.set_ylabel('Mean leakage score')
ax.set_title('Counterfactual leakage\\n(lower = less z_C species signal)')
ax.grid(True, alpha=0.3)

# Panel 2: species probe
ax = axes[1]
ax.plot(summary_df['gamma'], summary_df['species_probe_acc'],
        marker='o', color='darkorange', lw=2, label='z (26-dim)')
ax.axhline(1/N_SPECIES, color='red', ls=':', alpha=0.7, label=f'chance 1/{N_SPECIES}')
ax.set_xlabel('γ'); ax.set_ylabel('Species probe acc (z)')
ax.set_title('Species decodability from z\\n(lower = less species info in bottleneck)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 3: discriminability
ax = axes[2]
ax.plot(summary_df['gamma'], summary_df['mean_disc_acc'],
        marker='o', color='seagreen', lw=2)
ax.axhline(0.5, color='red', ls=':', alpha=0.7, label='chance 0.5')
ax.set_xlabel('γ'); ax.set_ylabel('Mean binary disc acc (z_C → species pair)')
ax.set_title('Binary discriminability in z_C\\n(lower = z_C less species-specific)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(
    'FunnyBirds MCBM: Three-level evidence for species-identity leakage vs γ\\n'
    '(All three measures should decrease as γ → IB suppresses concept backwash)',
    y=1.02,
)
plt.tight_layout()
plt.savefig('fb_mcbm_threelevel_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fb_mcbm_threelevel_vs_gamma.png')"""))

# ── Cell 36: correlation between measures ─────────────────────────────────────
cells.append(code("""\
# Correlation between the three measures across concepts (at gamma=0 and max gamma)
for g in [GAMMAS[0], GAMMAS[-1]]:
    agg  = concept_leakage[g]
    disc = (
        disc_results[g]
        .groupby('concept')['disc_acc'].mean()
        .reset_index(name='disc_acc')
    )
    prob = pd.DataFrame([
        {'concept': c, 'species_acc': a}
        for c, a in per_concept_probe[g].items()
    ])
    merged = (
        agg[['concept', 'part', 'leakage_mean']]
        .merge(disc, on='concept', how='inner')
        .merge(prob, on='concept', how='inner')
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (x_col, y_col, xl, yl) in zip(axes, [
        ('leakage_mean', 'disc_acc',   'Leakage score',  'Disc acc (z_C→pair)'),
        ('leakage_mean', 'species_acc','Leakage score',  'Species probe acc (z_C)'),
    ]):
        sub    = merged[[x_col, y_col, 'concept', 'part']].dropna()
        colors = [PART_COLORS.get(p, 'gray') for p in sub['part']]
        ax.scatter(sub[x_col], sub[y_col], c=colors, s=60, alpha=0.8, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row['concept'], (row[x_col], row[y_col]), fontsize=6, alpha=0.6)
        if len(sub) > 2:
            r, p = stats.pearsonr(sub[x_col], sub[y_col])
            m, b = np.polyfit(sub[x_col], sub[y_col], 1)
            xs   = np.array([sub[x_col].min(), sub[x_col].max()])
            ax.plot(xs, m * xs + b, 'k--', alpha=0.5, label=f'r={r:.2f}, p={p:.3f}')
            ax.legend(fontsize=8)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'FunnyBirds MCBM γ={g}: Cross-measure correlation', y=1.02)
    plt.tight_layout()
    plt.savefig(f'fb_mcbm_corr_gamma{g}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved fb_mcbm_corr_gamma{g}.png')"""))

# ── Cell 37: final summary markdown ──────────────────────────────────────────
cells.append(md("""\
## 10. Summary

### What we measured

| Measure | Tool | Interpretation |
|---|---|---|
| `leakage_sym` | Counterfactual swap | Causal: replacing z_C between species shifts species probability |
| `acc_full` | Species probe on z | Representational: 26-dim bottleneck encodes species |
| `disc_acc` | Binary probe on z_C | Mechanistic: single concept scalar separates two species |

### Reading the γ-sweep

- If all three measures **decrease with γ**: IB compression is suppressing the
  species-identity signal that leaks through the concept bottleneck — direct
  mechanistic support for the **concept backwash hypothesis**.
- If measures decrease but concept accuracy also drops: γ is too aggressive —
  the bottleneck is losing genuine concept information, not just the species identity.
- If no change: the IB penalty is not effective at this architecture/scale.

### Compare with CBM (γ=0 point)

`fb_mcbm_counterfactual_summary.csv` row `gamma=0.0` should closely match the values
from `funnybirds_cbm_counterfactual.ipynb` (same architecture, no IB)."""))

# ── Build notebook JSON ───────────────────────────────────────────────────────
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

out = Path("notebooks/funnybirds_mcbm_counterfactual.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")

# Quick sanity: parse back and count cells
nb2 = json.loads(out.read_text())
code_cells = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
md_cells   = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
print(f"Cells: {len(nb2['cells'])} total ({code_cells} code, {md_cells} markdown)")
print("Valid JSON: OK")
