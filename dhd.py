"""
Parallelised MCBM z-ordering sweep.

Speedups vs serial version:
  - N_RENDER_WORKERS threads fire renderer HTTP calls concurrently (I/O bound, safe)
  - GPU inference batched per-part instead of one image at a time
  - One gamma at a time (no multi-GPU risk / no OOM from loading multiple backbones)

Safety:
  - Per-render try/except: one bad render is skipped, not a crash
  - Renderer restart lock: concurrent threads don't race to restart Node.js
  - Per-part CSV saves: worst-case you lose <1 part on interrupt
  - Mini-batch GPU inference (batch_size=64): no OOM even on small GPUs

Usage:
    python run_z_ordering_sweep.py [--gammas 0.0 0.1 0.5] [--no_v2] [--force] [--workers 4]

SLURM header:
    #SBATCH --job-name=mcbm_sweep
    #SBATCH --output=logs/mcbm_sweep_%j.out
    #SBATCH --time=12:00:00
    #SBATCH --gres=gpu:1
    #SBATCH --mem=32G
    #SBATCH --cpus-per-task=6
    node funnybirds/render/server.js &
    sleep 5
    python run_z_ordering_sweep.py
"""

import argparse
import gc
import io
import itertools as _itools
import json
import random
import subprocess
import sys
import threading
import time
from base64 import decodebytes
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path('/scratch/network/cr7998/cv_emergence_project')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FB           = ROOT / 'data' / 'FunnyBirds'
RENDERER_DIR = ROOT.parent / 'funnybirds' / 'render'

assert FB.exists(), f'Missing FunnyBirds data: {FB}'

# ── CLI args ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--gammas',  nargs='+', type=float, default=[0.0, 0.1, 0.5, 1.0, 5.0])
parser.add_argument('--no_v2',  action='store_true')
parser.add_argument('--force',  action='store_true')
parser.add_argument('--workers', type=int, default=4,
                    help='Concurrent renderer threads (default 4; increase to 6-8 if renderer is stable)')
args = parser.parse_args()

GAMMAS            = args.gammas
USE_V2            = not args.no_v2
FORCE_RERUN       = args.force
N_RENDER_WORKERS  = args.workers
GPU_BATCH_SIZE    = 64   # images per GPU forward pass
N_SPECIES         = 50
MAX_IMGS_PER_SPECIES = 5
MAX_PAIRS_PER_PART   = 100
MCBM_Z_ACTIVE     =  3.0
MCBM_Z_INACTIVE   = -3.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}  workers={N_RENDER_WORKERS}  gpu_batch={GPU_BATCH_SIZE}')
print(f'GAMMAS={GAMMAS}  USE_V2={USE_V2}  FORCE_RERUN={FORCE_RERUN}')

def mcbm_ckpt(g):  return ROOT / 'checkpoints_funnybirds' / f'mcbm_fb_gamma{g}.pth'
def mcbm_feats(g): return ROOT / 'features' / f'resnet50_mcbm_fb_gamma{g}'

for g in GAMMAS:
    c, f = mcbm_ckpt(g), mcbm_feats(g)
    print(f'  gamma={g}  ckpt={c.exists()}  feats={f.exists()}')

# ── Data loading helpers ───────────────────────────────────────────────────────

def safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

def load_features(feat_dir, layer, split):
    p = feat_dir / f'{layer}_{split}.pt'
    assert p.exists(), f'Missing: {p}'
    X = safe_torch_load(p)
    return (X if isinstance(X, torch.Tensor) else torch.tensor(X)).float()

def to_1d_int_array(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x).reshape(-1).astype(int)

def load_split_order(feat_dir, split):
    p = feat_dir / f'labels_{split}.pt'
    t = safe_torch_load(p)
    assert isinstance(t, dict) and 'image_ids' in t
    return to_1d_int_array(t['image_ids'])

def load_species_maps(fb_root):
    df = pd.read_csv(fb_root / 'metadata' / 'classes.csv')
    return dict(zip(df['class_id'], df['class_name']))

def load_meta(fb_root):
    df = pd.read_csv(fb_root / 'metadata' / 'images.csv')
    df['species_id'] = df['class_id']
    df['species_name'] = df['class_id'].map(load_species_maps(fb_root))
    return df

# ── Concept / dataset setup ────────────────────────────────────────────────────

from datasets.funnybirds_dataset import FunnyBirdsDataset
from datasets.funnybirds_dataset import concept_names as _fb_concept_names
from datasets.funnybirds_dataset import _build_part_lookup
from datasets.funnybirds_dataset import PART_VARIANTS, _FUNNYBIRDS_N_TRAIN

CONCEPT_NAMES  = _fb_concept_names()
CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPT_NAMES)}
PARTS = ['beak', 'eye', 'wing', 'foot', 'tail']

_fb_ds    = FunnyBirdsDataset(FB, split='train')
cc_matrix = _fb_ds.get_class_concept_matrix()[0].numpy()

meta      = load_meta(FB)
_meta_idx = meta.set_index('image_id')

# ── Annotation / part parameters ──────────────────────────────────────────────

with open(FB / 'dataset_test.json') as f:
    test_anns = json.load(f)

with open(FB / 'parts.json') as f:
    parts_json = json.load(f)

parts_lookup     = _build_part_lookup(parts_json)
PARTS_WITH_COLOR = {p for p, vs in parts_json.items() if any('color' in v for v in vs)}

def variant_idx_from_ann(ann, part):
    model = ann.get(f'{part}_model', '')
    if not model or model == 'placeholder':
        return -1
    kf = {'model': model}
    if part in PARTS_WITH_COLOR:
        color = ann.get(f'{part}_color', '')
        if color:
            kf['color'] = color
    return parts_lookup[part].get(tuple(sorted(kf.items())), -1)

species_part_params = {}
species_variant_idx = {}
for ann in test_anns:
    sid = int(ann['class_idx'])
    if sid in species_part_params:
        continue
    species_part_params[sid] = {}
    species_variant_idx[sid] = {}
    for part in PARTS:
        params = {'model': ann.get(f'{part}_model', '')}
        if part in PARTS_WITH_COLOR:
            params['color'] = ann.get(f'{part}_color', '')
        species_part_params[sid][part] = params
        species_variant_idx[sid][part] = variant_idx_from_ann(ann, part)

test_idx_by_species = {sid: [] for sid in range(N_SPECIES)}
for local_idx, ann in enumerate(test_anns):
    test_idx_by_species[int(ann['class_idx'])].append(local_idx)

print(f'Loaded {len(test_anns)} test anns, {len(species_part_params)} species')

# ── Renderer ───────────────────────────────────────────────────────────────────

eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_server_proc          = None
_renderer_restart_lock = threading.Lock()   # prevents concurrent restarts

def json_to_url(sample, prefix='http://localhost:8081/render?', render_mode='default'):
    url = prefix + 'render_mode=' + render_mode + '&'
    for key in sample:
        if key != 'class_idx':
            url += key + '=' + str(sample[key]) + '&'
    return url[:-1]

def _raw_render(ann, render_mode='default'):
    url = json_to_url(ann, render_mode=render_mode)
    response = requests.get(url, timeout=30).content
    img = Image.open(io.BytesIO(decodebytes(response))).convert('RGB')
    resample = Image.NEAREST if render_mode == 'part_map' else Image.BILINEAR
    return img.resize((256, 256), resample=resample)

def check_renderer_alive(timeout=2.0):
    try:
        r = requests.get(
            'http://localhost:8081/render?render_mode=default'
            '&beak_model=beak01.glb&eye_model=eye01.glb'
            '&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red'
            '&wing_model=wing01.glb&wing_color=red'
            '&camera_distance=300&camera_pitch=0&camera_roll=0'
            '&light_distance=300&light_pitch=0&light_roll=0',
            timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def _restart_renderer():
    """Restart renderer server; serialised via lock so threads don't race."""
    global _server_proc
    with _renderer_restart_lock:
        if check_renderer_alive():
            return True   # another thread already restarted it
        try: _server_proc.kill()
        except: pass
        _server_proc = subprocess.Popen(
            ['node', 'server.js'], cwd=str(RENDERER_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(15):
            time.sleep(1)
            if check_renderer_alive():
                return True
        return False

def render_ann_safe(ann, max_retries=3):
    for attempt in range(max_retries):
        try:
            return _raw_render(ann, render_mode='default')
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            _restart_renderer()
            time.sleep(2)

def render_part_map_safe(ann, max_retries=3):
    for attempt in range(max_retries):
        try:
            return _raw_render(ann, render_mode='part_map')
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            _restart_renderer()
            time.sleep(2)

def start_renderer_server():
    global _server_proc
    if check_renderer_alive():
        print('[renderer] Already running.')
        return True
    if not (RENDERER_DIR / 'server.js').exists():
        print(f'[renderer] server.js not found at {RENDERER_DIR}')
        return False
    print('[renderer] Starting ...')
    _server_proc = subprocess.Popen(
        ['node', 'server.js'], cwd=str(RENDERER_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(15):
        time.sleep(1)
        if check_renderer_alive():
            print('[renderer] Up.')
            return True
    print('[renderer] Did not respond after 15s.')
    return False

RENDERER_AVAILABLE = start_renderer_server()
if not RENDERER_AVAILABLE:
    print('[FATAL] Renderer unavailable.'); sys.exit(1)

# ── Part-map segmentation ──────────────────────────────────────────────────────

PART_SEG_COLORS = {
    'beak': (255, 255,   0),
    'eye':  (255, 255, 253),
    'wing': (  0, 255,   1),
    'foot': (255,   0,   1),
    'tail': (  0,   0, 255),
}

def part_pixel_count(img_seg, part):
    arr = np.array(img_seg)
    r, g, b = PART_SEG_COLORS[part]
    return int(((arr[:,:,0]==r) & (arr[:,:,1]==g) & (arr[:,:,2]==b)).sum())

# ── MCBM loading ───────────────────────────────────────────────────────────────

def load_mcbm_weights(ckpt_path):
    ckpt = safe_torch_load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)
    return (sd['concept_encoder.weight'].float(),
            sd['concept_encoder.bias'].float(),
            sd['label_head.weight'].float(),
            sd['label_head.bias'].float(),
            ckpt.get('config', {}))

def load_mcbm_backbone(ckpt_path, device):
    ckpt = safe_torch_load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)
    prefix = 'backbone.'
    backbone_state = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(backbone_state, strict=False)
    return backbone.to(device).eval()

@torch.no_grad()
def compute_z_from_avgpool(avg, W_c, b_c):
    return avg @ W_c.T + b_c

# ── Part swap ─────────────────────────────────────────────────────────────────

def swap_part_in_ann(ann, part, new_params):
    cf = dict(ann)
    cf[f'{part}_model'] = new_params['model']
    if part in PARTS_WITH_COLOR:
        cf[f'{part}_color'] = new_params.get('color', '')
    return cf

# ── Species pairs ─────────────────────────────────────────────────────────────

rng = random.Random(42)   # fixed seed — must match notebook
all_pairs = {}
for part in PARTS:
    pairs = [
        (sid_A, species_variant_idx[sid_A][part],
         sid_B, species_variant_idx[sid_B][part])
        for sid_A, sid_B in combinations(range(N_SPECIES), 2)
        if species_variant_idx[sid_A][part] != species_variant_idx[sid_B][part]
    ]
    if len(pairs) > MAX_PAIRS_PER_PART:
        pairs = rng.sample(pairs, MAX_PAIRS_PER_PART)
    all_pairs[part] = pairs

total = sum(len(p) * 2 * MAX_IMGS_PER_SPECIES for p in all_pairs.values())
v2_mult = 2 if USE_V2 else 1
print(f'Total renders per gamma: {total * v2_mult}  '
      f'(~{total * v2_mult * 1.0 / N_RENDER_WORKERS / 60:.0f} min at 1s/render with {N_RENDER_WORKERS} workers)')

# ── Parallel render task ───────────────────────────────────────────────────────

def _render_task(task):
    """
    Thread worker: render one counterfactual image (+part_map if USE_V2).
    Returns (task_meta, img_cf, img_seg, error_str).
    Never raises — errors are returned as strings so the pool keeps running.
    """
    meta = task['meta']
    ann_cf = task['ann_cf']
    try:
        img_cf  = render_ann_safe(ann_cf)
        img_seg = render_part_map_safe(ann_cf) if USE_V2 else None
        return meta, img_cf, img_seg, None
    except Exception as e:
        return meta, None, None, str(e)

# ── Main sweep ────────────────────────────────────────────────────────────────

def _part_csv(g, part):
    suffix = '_v2' if USE_V2 else ''
    return Path(f'fb_mcbm_z_ordering_gamma{g}_{part}{suffix}.csv')

def _final_csv(g):
    suffix = '_v2' if USE_V2 else ''
    return Path(f'fb_mcbm_z_ordering_gamma{g}{suffix}.csv')


def run_z_ordering_for_gamma(g):
    ckpt_path  = mcbm_ckpt(g)
    feats_path = mcbm_feats(g)

    if not ckpt_path.exists():
        print(f'  [skip] gamma={g}: checkpoint not found'); return None
    if not feats_path.exists():
        print(f'  [skip] gamma={g}: features not found'); return None

    parts_needed = [p for p in PARTS if FORCE_RERUN or not _part_csv(g, p).exists()]
    if not parts_needed:
        print(f'  [cache] gamma={g}: all parts cached.')
        return pd.concat([pd.read_csv(_part_csv(g, p)) for p in PARTS], ignore_index=True)

    print(f'\n=== gamma={g} ===  parts to run: {parts_needed}')

    W_c, b_c, W_y, b_y, cfg = load_mcbm_weights(ckpt_path)
    backbone = load_mcbm_backbone(ckpt_path, device)
    W_c_d, b_c_d = W_c.to(device), b_c.to(device)
    W_y_d, b_y_d = W_y.to(device), b_y.to(device)

    ids_te    = load_split_order(feats_path, 'test')
    avg_te    = load_features(feats_path, 'avgpool', 'test')
    z_te      = compute_z_from_avgpool(avg_te, W_c, b_c)
    id_to_row = {int(i): r for r, i in enumerate(ids_te)}

    all_part_dfs = []

    for part in PARTS:
        part_csv = _part_csv(g, part)
        if part_csv.exists() and not FORCE_RERUN:
            print(f'  [cache] {part}')
            all_part_dfs.append(pd.read_csv(part_csv))
            continue

        # ── Build render tasks for whole part ─────────────────────────────────
        render_tasks = []
        for sid_A, var_A, sid_B, var_B in all_pairs[part]:
            for sid_src, var_src, sid_donor, var_donor, direction in [
                (sid_A, var_A, sid_B, var_B, 'fwd'),
                (sid_B, var_B, sid_A, var_A, 'bwd'),
            ]:
                for local_idx in test_idx_by_species[sid_src][:MAX_IMGS_PER_SPECIES]:
                    gid = _FUNNYBIRDS_N_TRAIN + local_idx
                    row_idx = id_to_row.get(gid)
                    if row_idx is None:
                        continue
                    ann_cf = swap_part_in_ann(
                        test_anns[local_idx], part, species_part_params[sid_donor][part])
                    render_tasks.append({
                        'ann_cf': ann_cf,
                        'meta': {
                            'sid_src':   sid_src,  'sid_donor': sid_donor,
                            'var_src':   var_src,  'var_donor': var_donor,
                            'direction': direction,
                            'z_orig':    z_te[row_idx],
                        },
                    })

        # ── Execute renders in parallel ────────────────────────────────────────
        render_results = []   # list of (meta, img_cf, img_seg)
        n_errors = 0
        with ThreadPoolExecutor(max_workers=N_RENDER_WORKERS) as pool:
            futures = {pool.submit(_render_task, t): i for i, t in enumerate(render_tasks)}
            for future in tqdm(as_completed(futures), total=len(render_tasks),
                               desc=f'  {part} renders'):
                meta, img_cf, img_seg, err = future.result()
                if err:
                    n_errors += 1
                    if n_errors <= 5:
                        print(f'  [WARN] render error (skipping): {err}')
                    continue
                render_results.append((meta, img_cf, img_seg))

        if n_errors:
            print(f'  [WARN] {part}: {n_errors}/{len(render_tasks)} renders failed, skipped.')

        # ── Batched GPU inference ──────────────────────────────────────────────
        rows = []
        print(f'  {part}: GPU inference on {len(render_results)} images (batch={GPU_BATCH_SIZE}) ...')

        for batch_start in range(0, len(render_results), GPU_BATCH_SIZE):
            batch = render_results[batch_start : batch_start + GPU_BATCH_SIZE]
            imgs  = torch.stack([eval_tf(img_cf) for _, img_cf, _ in batch])

            with torch.no_grad():
                avgs   = backbone(imgs.to(device))               # [B, 2048]
                z_cfs  = (avgs @ W_c_d.T + b_c_d).cpu()         # [B, 26]
                p_cfs  = torch.softmax(
                    z_cfs @ W_y.T + b_y, dim=-1)                 # [B, 50]

            for j, (meta, _, img_seg) in enumerate(batch):
                part_   = part
                z_orig  = meta['z_orig']
                sid_src = meta['sid_src'];  sid_donor = meta['sid_donor']
                var_src = meta['var_src'];  var_donor = meta['var_donor']

                c_src   = CONCEPT_TO_IDX[f'{part_}_{var_src}']
                c_donor = CONCEPT_TO_IDX[f'{part_}_{var_donor}']
                z_cf    = z_cfs[j]
                p_cf    = p_cfs[j]

                z_new  = float(z_cf[c_donor])
                z_old  = float(z_cf[c_src])
                margin = z_new - z_old

                z_gt          = z_orig.clone()
                z_gt[c_donor] = MCBM_Z_ACTIVE
                z_gt[c_src]   = MCBM_Z_INACTIVE
                p_gt = torch.softmax(z_gt @ W_y.T + b_y, dim=-1)

                row = {
                    'sid_src':          sid_src,
                    'sid_donor':        sid_donor,
                    'part':             part_,
                    'var_src':          var_src,
                    'var_donor':        var_donor,
                    'c_src':            c_src,
                    'c_donor':          c_donor,
                    'z_new':            z_new,
                    'z_old':            z_old,
                    'z_new_orig':       float(z_orig[c_donor]),
                    'z_old_orig':       float(z_orig[c_src]),
                    'margin':           margin,
                    'ordering_correct': bool(margin > 0),
                    'p_cf_donor':       float(p_cf[sid_donor]),
                    'p_gt_donor':       float(p_gt[sid_donor]),
                    'direction':        meta['direction'],
                }
                if USE_V2 and img_seg is not None:
                    row['pixel_count_cf'] = part_pixel_count(img_seg, part_)
                if part_ == 'tail':
                    for i in range(PART_VARIANTS['tail']):
                        row[f'z_cf_tail_{i}'] = float(z_cf[CONCEPT_TO_IDX[f'tail_{i}']])
                rows.append(row)

        part_df = pd.DataFrame(rows)
        part_df.to_csv(part_csv, index=False)
        fc = part_df['ordering_correct'].mean()
        mm = part_df['margin'].mean()
        print(f'    {part}: {len(part_df)} rows  ordering_correct={fc:.3%}  '
              f'mean_margin={mm:+.4f}  → saved {part_csv}')
        all_part_dfs.append(part_df)

    del backbone, W_c, b_c, W_y, b_y, W_c_d, b_c_d, W_y_d, b_y_d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.concat(all_part_dfs, ignore_index=True)


if __name__ == '__main__':
    for g in GAMMAS:
        final_csv = _final_csv(g)
        if final_csv.exists() and not FORCE_RERUN:
            print(f'[cache] gamma={g}: {final_csv} exists, skipping.')
            continue
        df_g = run_z_ordering_for_gamma(g)
        if df_g is not None:
            df_g.to_csv(final_csv, index=False)
            print(f'  Saved {final_csv}  ({len(df_g)} rows)')

    print('\nDone. In notebook: run from "## 12. Load all gamma results".')