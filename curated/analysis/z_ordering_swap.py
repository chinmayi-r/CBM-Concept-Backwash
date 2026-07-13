#!/usr/bin/env python3
# WIP PORT of ../../run_z_ordering_sweep.py to curated minimal_cbm. The renderer,
# part-swap, species-pairs, z-ordering record, and CSV logic port VERBATIM. Three
# contained changes are needed (each small; together a careful pass; untestable
# off-cluster -> iterate against the renderer via train/renderer_swap.slurm):
#
#  A. MODEL LOADING. Delete load_mcbm_weights / load_mcbm_backbone /
#     make_mcbm_inference_fns / compute_z_from_avgpool and the cached-feature reads
#     (load_features/load_split_order/avg_te/z_te). Replace with:
#         from grounding_deletion import load_model, _MEAN, _STD
#         model, n_concepts = load_model(config_basename, seed, epoch=None, device)
#     Map gamma -> config: config = f"funnybirds-mcbm-g{str(g).replace('.','p')}" ; loop --seeds.
#
#  B. CONCEPT/SPECIES MAPS. The driver imports datasets.funnybirds_dataset
#     (concept_names, _build_part_lookup, PART_VARIANTS, _FUNNYBIRDS_N_TRAIN) and uses
#     species_variant_idx / test_idx_by_species / species_part_params. Curated has all of
#     this in data/funnybirds/funnybirds_concepts.py + dataset_test.json params -> rewire
#     these to the curated module (sys.path -> curated/data/funnybirds).
#
#  C. THE z MAPPING (important). Old "z" = 26-d CONCEPT LOGITS (avg @ W_c + b_c). In
#     minimal_cbm that is out["c_logits"], NOT out["z"] (which is the pre-concept
#     bottleneck). So the inference fn returns c_logits for the margin
#     (z_new[c_donor]-z_old[c_src]) and out["y_preds"] for class prob. z_orig: compute
#     LIVE from the original render (render_ann_safe(ann_orig)) instead of cached feats.
#     The z2p_fn / p_gt_donor path does NOT map (minimal_cbm y-head reads the bottleneck,
#     not concept logits) -> drop p_gt_donor (secondary diagnostic) or reconstruct via
#     the model's concept intervention; keep p_cf_donor from out["y_preds"].
#
#  D. PATHS/ARGS: ROOT/FB/RENDERER_DIR -> args (--funnybirds-root --renderer-url --out);
#     write CSVs to <out>/  (=$CURATED_DATA/swap), not the repo root.
#
# Everything below is the ORIGINAL working driver, unmodified, for reference during the
# port. Do NOT run as-is (it targets the old code paths).

"""
Standalone sweep script: runs the MCBM z-ordering experiment for all gammas
and saves per-part + combined CSVs to disk.

After this finishes, open the notebook and run from cell
"## 12. Load all gamma results" onwards — it loads the CSVs directly.

Usage:
    python run_z_ordering_sweep.py [--gammas 0.0 0.1 0.5 1.0 5.0] [--no_v2] [--force] [--workers N]

SLURM example  (save as run_z_ordering_sweep.sh, then: sbatch run_z_ordering_sweep.sh):
    #!/bin/bash
    #SBATCH --job-name=mcbm_sweep
    #SBATCH --output=logs/mcbm_sweep_%j.out
    #SBATCH --error=logs/mcbm_sweep_%j.err
    #SBATCH --time=12:00:00
    #SBATCH --gres=gpu:1
    #SBATCH --mem=32G
    #SBATCH --cpus-per-task=6

    mkdir -p logs
    # conda is already active (cubvision-gpu) — no module load needed
    conda activate cubvision-gpu
    cd /scratch/network/cr7998/cv_emergence_project

    # Start renderer from its own directory
    RENDERER_DIR="/scratch/network/cr7998/funnybirds/render"
    cd "$RENDERER_DIR"
    node server.js > /tmp/renderer_${SLURM_JOB_ID}.log 2>&1 &
    RENDERER_PID=$!
    cd /scratch/network/cr7998/cv_emergence_project

    # Poll until renderer is up (max 30 s)
    RENDERER_UP=0
    for i in $(seq 1 30); do
        sleep 1
        curl -s --max-time 2 "http://localhost:8081/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" > /dev/null 2>&1 && RENDERER_UP=1 && break
        echo "  waiting for renderer... ${i}s"
    done
    if [ $RENDERER_UP -eq 0 ]; then
        echo "[ERROR] Renderer did not start. Node log:"
        cat /tmp/renderer_${SLURM_JOB_ID}.log
        kill $RENDERER_PID 2>/dev/null
        exit 1
    fi

    python run_z_ordering_sweep.py --gammas 0.0 0.1 0.5 1.0 5.0 --workers 4

    EXIT_CODE=$?
    kill $RENDERER_PID 2>/dev/null
    echo "Job finished: $(date)  exit_code=$EXIT_CODE"
    exit $EXIT_CODE
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
parser.add_argument('--gammas',      nargs='+', type=float, default=[0.0, 0.1, 0.5, 1.0, 5.0])
parser.add_argument('--no_v2',      action='store_true', help='Skip part_map renders (no pixel_count_cf)')
parser.add_argument('--force',      action='store_true', help='Re-run even if CSVs exist')
parser.add_argument('--workers',    type=int, default=4,
                    help='Parallel render workers (HTTP I/O only; GPU inference stays sequential)')
parser.add_argument('--ckpt_suffix', type=str, default='',
                    help='Suffix appended to checkpoint name, e.g. "_rl" for relabeled models. '
                         'Checkpoint: checkpoints_funnybirds/mcbm_fb_gamma{G}{suffix}.pth')
args = parser.parse_args()

GAMMAS               = args.gammas
USE_V2               = not args.no_v2
FORCE_RERUN          = args.force
N_RENDER_WORKERS     = args.workers
CKPT_SUFFIX          = args.ckpt_suffix
N_SPECIES            = 50
N_CONCEPTS           = 26
MAX_IMGS_PER_SPECIES = 5
MAX_PAIRS_PER_PART   = 100
MCBM_Z_ACTIVE        =  3.0
MCBM_Z_INACTIVE      = -3.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
print(f'GAMMAS={GAMMAS}  USE_V2={USE_V2}  FORCE_RERUN={FORCE_RERUN}  '
      f'N_RENDER_WORKERS={N_RENDER_WORKERS}  CKPT_SUFFIX={CKPT_SUFFIX!r}')

def mcbm_ckpt(g):  return ROOT / 'checkpoints_funnybirds' / f'mcbm_fb_gamma{g}{CKPT_SUFFIX}.pth'
def mcbm_feats(g): return ROOT / 'features' / f'resnet50_mcbm_fb_gamma{g}{CKPT_SUFFIX}'

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
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    return X.float()

def to_1d_int_array(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x).reshape(-1).astype(int)

def load_split_order(feat_dir, split):
    p = feat_dir / f'labels_{split}.pt'
    assert p.exists(), f'Missing: {p}'
    t = safe_torch_load(p)
    assert isinstance(t, dict) and 'image_ids' in t
    return to_1d_int_array(t['image_ids'])

def load_species_maps(fb_root):
    df = pd.read_csv(fb_root / 'metadata' / 'classes.csv')
    id2name = dict(zip(df['class_id'], df['class_name']))
    return id2name

def load_meta(fb_root):
    df = pd.read_csv(fb_root / 'metadata' / 'images.csv')
    id2name = load_species_maps(fb_root)
    df['species_id'] = df['class_id']
    df['species_name'] = df['class_id'].map(id2name)
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
cc_matrix, _ = _fb_ds.get_class_concept_matrix()
cc_matrix = cc_matrix.numpy()

id2name  = load_species_maps(FB)
meta     = load_meta(FB)
_meta_idx = meta.set_index('image_id')

# ── Annotation / part parameters ──────────────────────────────────────────────

ann_path_test = FB / 'dataset_test.json'
with open(ann_path_test) as f:
    test_anns = json.load(f)

with open(FB / 'parts.json') as f:
    parts_json = json.load(f)

parts_lookup     = _build_part_lookup(parts_json)
PARTS_WITH_COLOR = {
    part for part, variants in parts_json.items()
    if any('color' in v for v in variants)
}

def variant_idx_from_ann(ann, part):
    model = ann.get(f'{part}_model', '')
    if not model or model == 'placeholder':
        return -1
    key_fields = {'model': model}
    if part in PARTS_WITH_COLOR:
        color = ann.get(f'{part}_color', '')
        if color:
            key_fields['color'] = color
    return parts_lookup[part].get(tuple(sorted(key_fields.items())), -1)

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

print(f'Loaded {len(test_anns)} test annotations, {len(species_part_params)} species')

# ── Renderer ───────────────────────────────────────────────────────────────────

eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def json_to_url(sample, prefix='http://localhost:8081/render?', render_mode='default'):
    url = prefix + 'render_mode=' + render_mode + '&'
    for key in sample:
        if key == 'class_idx':
            continue
        url += key + '=' + str(sample[key]) + '&'
    return url[:-1]

def json_to_image(sample, mode='test'):
    url = json_to_url(sample, render_mode='part_map' if 'part_map' in mode else 'default')
    response = requests.get(url, timeout=30).content
    img = Image.open(io.BytesIO(decodebytes(response))).convert('RGB')
    resample = Image.NEAREST if 'part_map' in mode else Image.BILINEAR
    return img.resize((256, 256), resample=resample)

_server_proc          = None
_renderer_restart_lock = threading.Lock()

def render_ann_safe(ann, max_retries=3):
    """Render one annotation, restarting the renderer server if it dies.
    Thread-safe: only one thread at a time may restart the server."""
    for attempt in range(max_retries):
        try:
            return json_to_image(ann, mode='test')
        except Exception:
            if attempt == max_retries - 1:
                raise
            # Only one thread should restart the server at a time.
            with _renderer_restart_lock:
                # Re-check after acquiring the lock — another thread may have
                # already restarted it successfully.
                try:
                    return json_to_image(ann, mode='test')
                except Exception:
                    pass
                try:
                    _server_proc.kill()
                except Exception:
                    pass
                globals()['_server_proc'] = subprocess.Popen(
                    ['node', 'server.js'], cwd=str(RENDERER_DIR),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(6)

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

def start_renderer_server():
    global _server_proc
    if check_renderer_alive():
        print('[renderer] Server already running on port 8081.')
        return True
    if not (RENDERER_DIR / 'server.js').exists():
        print(f'[renderer] server.js not found at {RENDERER_DIR}')
        return False
    print(f'[renderer] Starting server from {RENDERER_DIR} ...')
    _server_proc = subprocess.Popen(
        ['node', 'server.js'], cwd=str(RENDERER_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(15):
        time.sleep(1)
        if check_renderer_alive():
            print('[renderer] Server is up.')
            return True
    print('[renderer] Server did not respond after 15s.')
    return False

RENDERER_AVAILABLE = start_renderer_server()
if not RENDERER_AVAILABLE:
    print('[FATAL] Renderer not available. Exiting.')
    sys.exit(1)

# ── Part-map / occlusion helpers ───────────────────────────────────────────────

PART_SEG_COLORS = {
    'beak': (255, 255,   0),
    'eye':  (255, 255, 253),
    'wing': (  0, 255,   1),
    'foot': (255,   0,   1),
    'tail': (  0,   0, 255),
}
_port_lock  = threading.Lock()
_port_cycle = _itools.cycle([8081])

def render_part_map(ann):
    with _port_lock:
        port = next(_port_cycle)
    url = json_to_url(ann, prefix=f'http://localhost:{port}/render?', render_mode='part_map')
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(decodebytes(resp.content))).convert('RGB').resize((256, 256), Image.NEAREST)

def part_pixel_count(img_seg, part):
    arr = np.array(img_seg)
    r, g, b = PART_SEG_COLORS[part]
    return int(((arr[:,:,0]==r) & (arr[:,:,1]==g) & (arr[:,:,2]==b)).sum())

# ── MCBM loading ───────────────────────────────────────────────────────────────

def load_mcbm_weights(ckpt_path):
    ckpt = safe_torch_load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)
    cfg  = ckpt.get('config', {})
    return (sd['concept_encoder.weight'].float(),
            sd['concept_encoder.bias'].float(),
            sd['label_head.weight'].float(),
            sd['label_head.bias'].float(),
            cfg)

def load_mcbm_backbone(ckpt_path, device):
    ckpt = safe_torch_load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)
    prefix = 'backbone.'
    backbone_state = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    print(f'  [backbone] {len(backbone_state)} keys from checkpoint '
          f'(expect ~230 for ResNet50; 0 = backbone NOT saved in ckpt!)')
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:     print(f'  [backbone] missing keys: {missing[:5]}{"..." if len(missing)>5 else ""}')
    if unexpected:  print(f'  [backbone] unexpected keys: {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
    return backbone.to(device).eval()

def make_mcbm_inference_fns(W_c, b_c, W_y, b_y, backbone, device):
    W_c_d, b_c_d = W_c.to(device), b_c.to(device)
    W_y_d, b_y_d = W_y.to(device), b_y.to(device)

    @torch.no_grad()
    def run_through_mcbm(img_pil):
        x   = eval_tf(img_pil).unsqueeze(0).to(device)
        avg = backbone(x)
        z   = avg @ W_c_d.T + b_c_d
        p   = torch.softmax(z @ W_y_d.T + b_y_d, dim=-1)
        return z.squeeze(0).cpu(), p.squeeze(0).cpu()

    @torch.no_grad()
    def z_to_probs_mcbm(z_raw):
        return torch.softmax(z_raw @ W_y.T + b_y, dim=-1)

    return run_through_mcbm, z_to_probs_mcbm

@torch.no_grad()
def compute_z_from_avgpool(avg, W_c, b_c):
    return avg @ W_c.T + b_c

# ── Part swap helpers ──────────────────────────────────────────────────────────

def swap_part_in_ann(ann, part, new_params):
    cf = dict(ann)
    cf[f'{part}_model'] = new_params['model']
    if part in PARTS_WITH_COLOR:
        cf[f'{part}_color'] = new_params.get('color', '')
    return cf

# ── Species pairs (fixed seed — must match notebook) ──────────────────────────

rng = random.Random(42)
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
print(f'Pairs defined. Total renders per gamma: {total}  (~{total*1.5/60:.0f} min at 1.5s each, '
      f'~{total*1.5/60/N_RENDER_WORKERS:.0f} min with {N_RENDER_WORKERS} parallel workers)')

# ── Per-part CSV helpers ───────────────────────────────────────────────────────

def _part_csv(g, part, use_v2):
    v2 = '_v2' if use_v2 else ''
    return Path(f'fb_mcbm_z_ordering_gamma{g}{CKPT_SUFFIX}_{part}{v2}.csv')

def _final_csv(g, use_v2):
    v2 = '_v2' if use_v2 else ''
    return Path(f'fb_mcbm_z_ordering_gamma{g}{CKPT_SUFFIX}{v2}.csv')

# ── Main sweep ─────────────────────────────────────────────────────────────────

def run_z_ordering_for_gamma(g, use_v2=False):
    ckpt_path  = mcbm_ckpt(g)
    feats_path = mcbm_feats(g)

    if not ckpt_path.exists():
        print(f'  [skip] gamma={g}: checkpoint not found')
        return None
    if not feats_path.exists():
        print(f'  [skip] gamma={g}: features not found')
        return None

    parts_needed = [p for p in PARTS if FORCE_RERUN or not _part_csv(g, p, use_v2).exists()]
    if not parts_needed:
        print(f'  [cache] gamma={g}: all parts done, loading from part CSVs.')
        return pd.concat([pd.read_csv(_part_csv(g, p, use_v2)) for p in PARTS], ignore_index=True)

    print(f'\n=== gamma={g} ===')
    print(f'  Parts to run: {parts_needed}')

    W_c, b_c, W_y, b_y, cfg = load_mcbm_weights(ckpt_path)
    backbone = load_mcbm_backbone(ckpt_path, device)
    run_fn, z2p_fn = make_mcbm_inference_fns(W_c, b_c, W_y, b_y, backbone, device)

    ids_te    = load_split_order(feats_path, 'test')
    avg_te    = load_features(feats_path, 'avgpool', 'test')
    z_te      = compute_z_from_avgpool(avg_te, W_c, b_c)
    id_to_row = {int(i): r for r, i in enumerate(ids_te)}

    # ── Sanity check: z_te should vary across images ──────────────────────────
    z_std = z_te.std(dim=0)   # [26] per-concept std across test set
    print(f'  [diag] z_te std (first 5 concepts): {z_std[:5].tolist()}')
    print(f'  [diag] avg_te std: {avg_te.std().item():.4f}  '
          f'W_c abs-mean: {W_c.abs().mean().item():.4f}  '
          f'b_c range: [{b_c.min().item():.3f}, {b_c.max().item():.3f}]')

    all_part_dfs = []

    for part in PARTS:
        part_csv = _part_csv(g, part, use_v2)
        if part_csv.exists() and not FORCE_RERUN:
            print(f'  [cache] {part}: {part_csv}')
            all_part_dfs.append(pd.read_csv(part_csv))
            continue

        # ── Build the full job list for this part ──────────────────────────
        # Each job is a dict with all metadata needed for both render and inference.
        jobs = []
        for sid_A, var_A, sid_B, var_B in all_pairs[part]:
            for local_idx in test_idx_by_species[sid_A][:MAX_IMGS_PER_SPECIES]:
                gid = _FUNNYBIRDS_N_TRAIN + local_idx
                row_idx = id_to_row.get(gid)
                if row_idx is None:
                    continue
                ann_orig = test_anns[local_idx]
                jobs.append({
                    'ann_orig': ann_orig,
                    'ann_cf':   swap_part_in_ann(ann_orig, part, species_part_params[sid_B][part]),
                    'sid_src':  sid_A, 'var_src':   var_A,
                    'sid_donor':sid_B, 'var_donor':  var_B,
                    'z_orig':   z_te[row_idx],
                    'direction':'fwd',
                })
            for local_idx in test_idx_by_species[sid_B][:MAX_IMGS_PER_SPECIES]:
                gid = _FUNNYBIRDS_N_TRAIN + local_idx
                row_idx = id_to_row.get(gid)
                if row_idx is None:
                    continue
                ann_orig = test_anns[local_idx]
                jobs.append({
                    'ann_orig': ann_orig,
                    'ann_cf':   swap_part_in_ann(ann_orig, part, species_part_params[sid_A][part]),
                    'sid_src':  sid_B, 'var_src':   var_B,
                    'sid_donor':sid_A, 'var_donor':  var_A,
                    'z_orig':   z_te[row_idx],
                    'direction':'bwd',
                })

        # ── Phase 1: parallel renders (I/O-bound HTTP, safe for threads) ──
        # Results stored in order so Phase 2 can stay sequential.
        render_results = [None] * len(jobs)

        def _render_job(idx):
            job = jobs[idx]
            img_cf  = render_ann_safe(job['ann_cf'])
            img_seg = render_part_map(job['ann_cf']) if use_v2 else None
            return idx, img_cf, img_seg

        with ThreadPoolExecutor(max_workers=N_RENDER_WORKERS) as pool:
            futures = {pool.submit(_render_job, i): i for i in range(len(jobs))}
            for fut in tqdm(as_completed(futures), total=len(jobs),
                            desc=f'  {part} render'):
                idx, img_cf, img_seg = fut.result()
                render_results[idx] = (img_cf, img_seg)

        # ── Phase 2: sequential per-image GPU inference ────────────────────
        # One image at a time through backbone (avoids BatchNorm batch effects).
        rows = []
        for i, job in enumerate(tqdm(jobs, desc=f'  {part} infer')):
            img_cf, img_seg = render_results[i]

            c_src   = CONCEPT_TO_IDX[f'{part}_{job["var_src"]}']
            c_donor = CONCEPT_TO_IDX[f'{part}_{job["var_donor"]}']

            z_cf, p_cf = run_fn(img_cf)   # single-image; no BatchNorm artefacts

            z_new  = float(z_cf[c_donor])
            z_old  = float(z_cf[c_src])
            margin = z_new - z_old

            z_gt          = job['z_orig'].clone()
            z_gt[c_donor] = MCBM_Z_ACTIVE
            z_gt[c_src]   = MCBM_Z_INACTIVE
            p_gt          = z2p_fn(z_gt)

            row = {
                'sid_src':   job['sid_src'],   'sid_donor': job['sid_donor'],
                'part':      part,
                'var_src':   job['var_src'],   'var_donor': job['var_donor'],
                'c_src':     c_src,            'c_donor':   c_donor,
                'z_new':     z_new,            'z_old':     z_old,
                'z_new_orig': float(job['z_orig'][c_donor]),
                'z_old_orig': float(job['z_orig'][c_src]),
                'margin':    margin,
                'ordering_correct': bool(margin > 0),
                'p_cf_donor': float(p_cf[job['sid_donor']]),
                'p_gt_donor': float(p_gt[job['sid_donor']]),
                'direction': job['direction'],
            }
            if use_v2:
                row['pixel_count_cf'] = part_pixel_count(img_seg, part)
            if part == 'tail':
                for ti in range(PART_VARIANTS['tail']):
                    row[f'z_cf_tail_{ti}'] = float(z_cf[CONCEPT_TO_IDX[f'tail_{ti}']])
            rows.append(row)

        # Free render images before moving to next part
        del render_results

        part_df = pd.DataFrame(rows)

        # ── Variance diagnostic: if z_new/z_old have ~0 std, backbone isn't varying ──
        z_new_std = part_df['z_new'].std()
        z_old_std = part_df['z_old'].std()
        fwd_margins = part_df.loc[part_df['direction']=='fwd', 'margin']
        bwd_margins = part_df.loc[part_df['direction']=='bwd', 'margin']
        print(f'  [diag] {part}: z_new std={z_new_std:.4f}  z_old std={z_old_std:.4f}  '
              f'fwd mean={fwd_margins.mean():+.4f}  bwd mean={bwd_margins.mean():+.4f}')

        part_df.to_csv(part_csv, index=False)
        fwd_df   = part_df[part_df['direction'] == 'fwd']
        bwd_df   = part_df[part_df['direction'] == 'bwd']
        fwd_acc  = fwd_df['ordering_correct'].mean()
        bwd_acc  = bwd_df['ordering_correct'].mean()
        fwd_mean = fwd_df['margin'].mean()
        bwd_mean = bwd_df['margin'].mean()
        print(f'    {part}: {len(part_df)} rows  '
              f'fwd_acc={fwd_acc:.3%} (mean={fwd_mean:+.4f})  '
              f'bwd_acc={bwd_acc:.3%} (mean={bwd_mean:+.4f})  '
              f'→ saved {part_csv}')
        all_part_dfs.append(part_df)

    del backbone, W_c, b_c, W_y, b_y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.concat(all_part_dfs, ignore_index=True)


if __name__ == '__main__':
    for g in GAMMAS:
        final_csv = _final_csv(g, USE_V2)
        if final_csv.exists() and not FORCE_RERUN:
            print(f'[cache] gamma={g}: {final_csv} already exists, skipping.')
            continue
        df_g = run_z_ordering_for_gamma(g, use_v2=USE_V2)
        if df_g is not None:
            df_g.to_csv(final_csv, index=False)
            print(f'  Saved {final_csv}  ({len(df_g)} rows)')

    print('\nDone. Load results in notebook from cell "## 12. Load all gamma results".')
