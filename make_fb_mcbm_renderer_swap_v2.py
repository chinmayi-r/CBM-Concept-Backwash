#!/usr/bin/env python3
"""
Generate funnybird_notebooks/fb_mcbm_renderer_swap_v2.ipynb

Mirrors fb_cbm_renderer_swap_v2_parallel.ipynb exactly, with MCBM adaptations:
  - No sigmoid in z computation (raw concept logits)
  - checkpoint key: concept_encoder.weight/bias (not concept_head)
  - GT ceiling: ±3 (IB q_phi saturation range)
  - Full gamma sweep [0.0, 0.1, 0.5, 1.0, 5.0]
  - Cross-gamma comparison plots at end
"""

import json, textwrap
from pathlib import Path

def cell(source, cell_type="code"):
    src_lines = source.split("\n")
    src_json  = [ln + "\n" for ln in src_lines[:-1]] + [src_lines[-1]]
    return {"cell_type": cell_type,
            "metadata": {},
            "source": src_json,
            **({} if cell_type == "markdown" else
               {"execution_count": None, "outputs": []})}

def md(src): return cell(src, "markdown")
def code(src): return cell(src.strip(), "code")

# ── cells ────────────────────────────────────────────────────────────────────
cells = []

# 0: Title
cells.append(md("""\
# FunnyBirds MCBM — Renderer-Based Part Swap (Gamma Sweep v2)

**Replicates `fb_cbm_renderer_swap_v2_parallel.ipynb` for MCBM across all trained γ values.**

**MCBM vs CBM key differences:**
- Checkpoint keys: `concept_encoder.weight/bias` (not `concept_head`)
- z is *raw* (no sigmoid): `z_raw = avgpool @ W_c.T + b_c ∈ ℝ`; label_head takes z_raw directly
- GT ceiling: z_raw = +3 (active) / −3 (inactive), matching IB q_phi saturation range
- γ = 0.0 → plain MCBM (sanity-check vs CBM); higher γ → stronger IB → expect less backwash

**Three measurements per swap:**
1. **z-ordering** — after inserting donor part, does z[donor_dim] > z[src_dim]?
2. **Margin** — z_cf[c_donor] − z_cf[c_src]  (positive = correct ordering)
3. **Leakage** — P_gt[donor] − P_render[donor]  (0 = perfect visual grounding)

**Gammas:** [0.0, 0.1, 0.5, 1.0, 5.0]"""))

# 1: Imports
cells.append(code("""\
import gc
import io
import json
import random
import subprocess
import sys
import time
from base64 import decodebytes
from itertools import combinations
from pathlib import Path

import requests
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import torch
import torch.nn as nn
from torchvision import models, transforms
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm"""))

# 2: Paths md
cells.append(md("## 0. Paths and configuration"))

# 3: Config
cells.append(code("""\
ROOT = Path('/scratch/network/cr7998/cv_emergence_project')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FB           = ROOT / 'data' / 'FunnyBirds'
RENDERER_DIR = ROOT.parent / 'funnybirds' / 'render'

GAMMAS = [0.0, 0.1, 0.5, 1.0, 5.0]

def mcbm_ckpt(g):  return ROOT / 'checkpoints_funnybirds' / f'mcbm_fb_gamma{g}.pth'
def mcbm_feats(g): return ROOT / 'features' / f'resnet50_mcbm_fb_gamma{g}'

assert FB.exists(), f'Missing FunnyBirds data: {FB}'
for g in GAMMAS:
    c, f = mcbm_ckpt(g), mcbm_feats(g)
    print(f'  gamma={g}  ckpt={c.exists()}  feats={f.exists()}')

N_SPECIES  = 50
N_CONCEPTS = 26

# ── Experiment limits ─────────────────────────────────────────────────────────
MAX_IMGS_PER_SPECIES = 5
MAX_PAIRS_PER_PART   = 100

# GT ceiling constants (IB q_phi saturation range is [-3, 3])
MCBM_Z_ACTIVE   =  3.0
MCBM_Z_INACTIVE = -3.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
print(f'MAX_IMGS_PER_SPECIES={MAX_IMGS_PER_SPECIES}  MAX_PAIRS_PER_PART={MAX_PAIRS_PER_PART}')"""))

# 4: Data helpers md
cells.append(md("## 1. Data loading helpers  *(from fb_recallv2.py)*"))

# 5: Loading helpers
cells.append(code("""\
def load_species_maps(fb_root: Path):
    classes_csv = fb_root / 'metadata' / 'classes.csv'
    if not classes_csv.exists():
        raise FileNotFoundError(
            'metadata/classes.csv not found. Run prepare_funnybirds_metadata.py first.')
    df = pd.read_csv(classes_csv)
    id2name  = dict(zip(df['class_id'], df['class_name']))
    id2short = {k: v.replace('funnybird_', 'FB') for k, v in id2name.items()}
    return id2name, id2short


def load_meta(fb_root: Path) -> pd.DataFrame:
    images_csv = fb_root / 'metadata' / 'images.csv'
    if not images_csv.exists():
        raise FileNotFoundError(
            'metadata/images.csv not found. Run prepare_funnybirds_metadata.py first.')
    df = pd.read_csv(images_csv)
    id2name, _ = load_species_maps(fb_root)
    df['species_id']   = df['class_id']
    df['species_name'] = df['class_id'].map(id2name)
    return df


print('Defined: load_species_maps  load_meta')"""))

# 6: Feature loading helpers
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


def load_split_order(feat_dir: Path, split: str):
    p = feat_dir / f'labels_{split}.pt'
    assert p.exists(), f'Missing: {p}'
    t = safe_torch_load(p)
    assert isinstance(t, dict), f'Expected dict in {p}, got {type(t)}'
    assert 'image_ids' in t, f'{p} missing image_ids; has {list(t.keys())}'
    return to_1d_int_array(t['image_ids'])


print('Defined: safe_torch_load  load_features  load_split_order')"""))

# 7: Concept names, cc_matrix md
cells.append(md("## 2. Concept names, class-concept matrix, metadata"))

# 8: Concept setup
cells.append(code("""\
from datasets.funnybirds_dataset import FunnyBirdsDataset
from datasets.funnybirds_dataset import concept_names as _fb_concept_names
from datasets.funnybirds_dataset import _build_part_lookup, _params_to_variant_idx
from datasets.funnybirds_dataset import PART_VARIANTS, _FUNNYBIRDS_N_TRAIN

CONCEPT_NAMES  = _fb_concept_names()   # 26 strings: ['beak_0', ..., 'tail_8']
CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPT_NAMES)}

PARTS = ['beak', 'eye', 'wing', 'foot', 'tail']
PART_GROUPS = {
    part: [i for i, c in enumerate(CONCEPT_NAMES) if c.startswith(f'{part}_')]
    for part in PARTS
}
PART_COLORS = {
    'beak': 'steelblue', 'eye': 'purple', 'wing': 'seagreen',
    'foot': 'darkorange', 'tail': 'crimson',
}
CONCEPT_TO_PART = {
    c: part
    for part, idxs in PART_GROUPS.items()
    for c in [CONCEPT_NAMES[i] for i in idxs]
}

# Exact class-concept matrix (no annotation noise)
_fb_ds    = FunnyBirdsDataset(FB, split='train')
cc_matrix, _ = _fb_ds.get_class_concept_matrix()   # [50, 26] int tensor
cc_matrix = cc_matrix.numpy()

print(f'Concepts ({len(CONCEPT_NAMES)}): {CONCEPT_NAMES}')
print(f'CC matrix: {cc_matrix.shape}  (each row sums to {cc_matrix.sum(1).mean():.0f})')"""))

# 9: Metadata
cells.append(code("""\
meta       = load_meta(FB)
id2name, _ = load_species_maps(FB)
def spname(sid: int) -> str:
    return id2name.get(int(sid), f'funnybird_{int(sid):02d}')

_meta_idx = meta.set_index('image_id')
print(f'meta: {len(meta)} images  ({meta["is_train"].sum()} train, {(meta["is_train"]==0).sum()} test)')"""))

# 10: MCBM loading md
cells.append(md("""\
## 3. MCBM loading helpers

Key differences from CBM:
- Checkpoint key: `concept_encoder.weight/bias` (not `concept_head`)
- `z_raw = avgpool @ W_c.T + b_c`  — **no sigmoid**; label_head takes z_raw directly
- GT ceiling uses ±3 (IB q_phi saturation range) instead of 0/1"""))

# 11: MCBM weights loader
cells.append(code("""\
def load_mcbm_weights(ckpt_path: Path):
    \"\"\"Return (W_c, b_c, W_y, b_y, config) from MCBM checkpoint.\"\"\"
    ckpt = safe_torch_load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)
    cfg  = ckpt.get('config', {})
    W_c  = sd['concept_encoder.weight'].float()   # [26, 2048]
    b_c  = sd['concept_encoder.bias'].float()     # [26]
    W_y  = sd['label_head.weight'].float()        # [50, 26]
    b_y  = sd['label_head.bias'].float()          # [50]
    return W_c, b_c, W_y, b_y, cfg


def load_mcbm_backbone(ckpt_path: Path, device):
    \"\"\"Extract backbone from MCBM checkpoint; return eval-mode ResNet50.\"\"\"
    ckpt = safe_torch_load(ckpt_path)
    sd   = ckpt.get('model_state_dict', ckpt)
    backbone_sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(backbone_sd, strict=True)
    return backbone.to(device).eval()


@torch.no_grad()
def compute_z_mcbm(avgpool: torch.Tensor, W_c: torch.Tensor, b_c: torch.Tensor) -> torch.Tensor:
    \"\"\"z_raw = avgpool @ W_c.T + b_c  — NO sigmoid.  Shape [N, 26].\"\"\"
    return avgpool @ W_c.T + b_c


print('Defined: load_mcbm_weights  load_mcbm_backbone  compute_z_mcbm')"""))

# 12: Checkpoint sanity check
cells.append(md("### 3a. Checkpoint sanity check — verify stored γ matches filename"))

cells.append(code("""\
import datetime

print(f'{"gamma":>8s}  {"ckpt_exists":>12s}  {"stored_gamma":>13s}  {"match":>6s}  '
      f'{"sigma":>6s}  {"lambda_c":>9s}  mtime')
print('-' * 80)
for g in GAMMAS:
    p = mcbm_ckpt(g)
    if not p.exists():
        print(f'{g:>8.1f}  {"MISSING":>12s}')
        continue
    try:
        ckpt      = safe_torch_load(p)
        cfg       = ckpt.get('config', {})
        stored_g  = cfg.get('gamma', 'N/A')
        stored_s  = cfg.get('sigma', 'N/A')
        stored_lc = cfg.get('lambda_c', 'N/A')
        match     = '✓' if abs(float(stored_g) - g) < 1e-6 else '✗ MISMATCH'
        mtime     = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        print(f'{g:>8.1f}  {"OK":>12s}  {stored_g!s:>13s}  {match:>6s}  '
              f'{str(stored_s):>6s}  {str(stored_lc):>9s}  {mtime}')
    except Exception as e:
        print(f'{g:>8.1f}  ERROR loading: {e}')
print()
print('NOTE: If stored_gamma != filename gamma the model was NOT retrained with that gamma.')"""))

# 14: FunnyBirds rendering parameters md
cells.append(md("## 4. Load FunnyBirds rendering parameters"))

# 15: test_anns
cells.append(code("""\
ann_path_test = FB / 'dataset_test.json'
assert ann_path_test.exists(), (
    f'{ann_path_test} not found.\\n'
    'Download: wget https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip')

with open(ann_path_test) as f:
    test_anns = json.load(f)

N_TEST = len(test_anns)
print(f'Loaded {N_TEST} test annotations')
print(f'Fields in one entry: {list(test_anns[0].keys())}')"""))

# 16: parts.json
cells.append(code("""\
parts_path = FB / 'parts.json'
assert parts_path.exists(), f'parts.json not found at {parts_path}'

with open(parts_path) as f:
    parts_json = json.load(f)

parts_lookup = _build_part_lookup(parts_json)

PARTS_WITH_COLOR = set()
for part, variants in parts_json.items():
    if any('color' in v for v in variants):
        PARTS_WITH_COLOR.add(part)

print(f'Parts with color field: {PARTS_WITH_COLOR}')
print('Variant counts per part:')
for part, variants in parts_json.items():
    print(f'  {part:6s}: {len(variants)} variants')


def variant_idx_from_ann(ann: dict, part: str) -> int:
    model = ann.get(f'{part}_model', '')
    if not model or model == 'placeholder':
        return -1
    key_fields = {'model': model}
    if part in PARTS_WITH_COLOR:
        color = ann.get(f'{part}_color', '')
        if color:
            key_fields['color'] = color
    key = tuple(sorted(key_fields.items()))
    return parts_lookup[part].get(key, -1)"""))

# 17: species_part_params
cells.append(code("""\
species_part_params = {}   # sid -> part -> {'model': str, 'color': str}
species_variant_idx = {}   # sid -> part -> variant_index (int)

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
        species_part_params[sid][part]  = params
        species_variant_idx[sid][part]  = variant_idx_from_ann(ann, part)

print(f'Extracted part params for {len(species_part_params)} species')
print(f'Species 0 parts: {species_part_params[0]}')"""))

# 18: tail variant distribution
cells.append(code("""\
tail_var_to_sids = {}
for sid in range(N_SPECIES):
    v = species_variant_idx[sid]['tail']
    tail_var_to_sids.setdefault(v, []).append(sid)

print('Tail variant distribution:')
for v, sids in sorted(tail_var_to_sids.items()):
    print(f'  tail_{v}: {len(sids)} species → {[spname(s) for s in sids[:4]]}...')

n_diff_pairs = sum(
    1 for a, b in combinations(range(N_SPECIES), 2)
    if species_variant_idx[a]['tail'] != species_variant_idx[b]['tail']
)
print(f'\\nTail-differing species pairs: {n_diff_pairs} / {N_SPECIES*(N_SPECIES-1)//2} total')"""))

# 19: Renderer setup md
cells.append(md("""\
## 5. Renderer setup (FunnyBirds Node.js server)

The FunnyBirds renderer runs as a local Node.js HTTP server on port 8081.

**Setup (once):**
```bash
git clone https://github.com/visinf/funnybirds.git
cd funnybirds/render
npm install
node server.js   # keep running while notebook executes
```"""))

# 20: renderer check
cells.append(code("""\
_server_js = RENDERER_DIR / 'server.js'
if not _server_js.exists():
    print('[WARNING] FunnyBirds renderer not found at:', RENDERER_DIR)
    print('Clone it with:')
    print('  git clone https://github.com/visinf/funnybirds.git')
    print('  cd funnybirds/render && npm install')
    print('Then update RENDERER_DIR in Cell 3 and re-run.')
    print()
    print('All cells up to the main loop will still run (for inspection).')
    print('Renderer-dependent cells are guarded by RENDERER_AVAILABLE.')
else:
    print(f'[OK] Renderer found at {RENDERER_DIR}')"""))

# 21: json_to_url, render functions
cells.append(code("""\
# Copied verbatim from visinf/funnybirds/render/create_dataset.py
def json_to_url(sample: dict,
                prefix: str = 'http://localhost:8081/render?',
                render_mode: str = 'default') -> str:
    url = prefix + 'render_mode=' + render_mode + '&'
    for key in list(sample.keys()):
        if key == 'class_idx':
            continue
        url = url + key + '=' + str(sample[key]) + '&'
    return url[:-1]


def json_to_image(sample: dict, mode: str = 'test') -> Image.Image:
    if mode in ('train', 'test'):
        url = json_to_url(sample)
    elif mode in ('train_part_map', 'test_part_map'):
        url = json_to_url(sample, render_mode='part_map')
    else:
        raise NotImplementedError(f'Unknown mode: {mode}')
    response = requests.get(url, timeout=30).content
    img_bytes = decodebytes(response)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    resample = Image.NEAREST if 'part_map' in mode else Image.BILINEAR
    return img.resize((256, 256), resample=resample)


_server_proc = None

def render_ann_safe(ann: dict, max_retries: int = 3) -> Image.Image:
    for attempt in range(max_retries):
        try:
            return json_to_image(ann, mode='test')
        except Exception:
            if attempt == max_retries - 1:
                raise
            print(f'[renderer] crash detected, restarting (attempt {attempt+1})...')
            try: _server_proc.kill()
            except: pass
            globals()['_server_proc'] = subprocess.Popen(
                ['node', 'server.js'], cwd=str(RENDERER_DIR),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(6)

def render_ann(ann): return render_ann_safe(ann)
print('Defined: json_to_url  json_to_image  render_ann')"""))

# 22: start_renderer_server
cells.append(code("""\
def check_renderer_alive(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(
            'http://localhost:8081/render?render_mode=default'
            '&beak_model=beak01.glb&eye_model=eye01.glb'
            '&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red'
            '&wing_model=wing01.glb&wing_color=red'
            '&camera_distance=300&camera_pitch=0&camera_roll=0'
            '&light_distance=300&light_pitch=0&light_roll=0',
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def start_renderer_server(renderer_dir: Path) -> bool:
    global _server_proc
    if check_renderer_alive():
        print('[renderer] Server already running on port 8081.')
        return True
    if not (renderer_dir / 'server.js').exists():
        print(f'[renderer] server.js not found at {renderer_dir}')
        return False
    print(f'[renderer] Starting server: node server.js in {renderer_dir} ...')
    _server_proc = subprocess.Popen(
        ['node', 'server.js'],
        cwd=str(renderer_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(15):
        time.sleep(1)
        if check_renderer_alive():
            print('[renderer] Server is up.')
            return True
    print('[renderer] Server did not respond after 15 s. Check Node.js logs.')
    return False


RENDERER_AVAILABLE = start_renderer_server(RENDERER_DIR)
print(f'RENDERER_AVAILABLE = {RENDERER_AVAILABLE}')"""))

# 23: eval_tf + run_through_mcbm factory
cells.append(code("""\
eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def make_mcbm_runner(W_c, b_c, W_y, b_y, backbone, device):
    \"\"\"
    Returns (run_through_mcbm, z_to_probs_mcbm) closures.
    z_raw = backbone(img) @ W_c.T + b_c   (no sigmoid)
    probs = softmax(z_raw @ W_y.T + b_y)
    \"\"\"
    W_c_d = W_c.to(device); b_c_d = b_c.to(device)
    W_y_d = W_y.to(device); b_y_d = b_y.to(device)

    @torch.no_grad()
    def run_through_mcbm(img_pil: Image.Image):
        x   = eval_tf(img_pil).unsqueeze(0).to(device)
        avg = backbone(x)
        z   = avg @ W_c_d.T + b_c_d                       # [1, 26] raw — no sigmoid
        p   = torch.softmax(z @ W_y_d.T + b_y_d, dim=-1)
        return z.squeeze(0).cpu(), p.squeeze(0).cpu()

    @torch.no_grad()
    def z_to_probs_mcbm(z_raw: torch.Tensor) -> torch.Tensor:
        return torch.softmax(z_raw @ W_y_d.cpu().T + b_y_d.cpu(), dim=-1)

    return run_through_mcbm, z_to_probs_mcbm

print('Defined: eval_tf  make_mcbm_runner')"""))

# 24: Parallel renderer setup md
cells.append(md("## 5a. Parallel renderer (4 ports)"))

# 25: parallel ports
cells.append(code("""\
import threading, itertools as _itools

RENDERER_PORTS = [8081, 8082, 8083, 8084]
_renderer_procs = {8081: _server_proc}

for port in RENDERER_PORTS[1:]:
    env = {**__import__('os').environ, 'PORT': str(port)}
    p = subprocess.Popen(['node', 'server.js'], cwd=str(RENDERER_DIR),
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _renderer_procs[port] = p

time.sleep(6)

alive_ports = []
for port in RENDERER_PORTS:
    try:
        r = requests.get(
            f'http://localhost:{port}/render?render_mode=default'
            '&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb'
            '&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red'
            '&camera_distance=300&camera_pitch=0&camera_roll=0'
            '&light_distance=300&light_pitch=0&light_roll=0', timeout=20)
        if r.status_code == 200:
            alive_ports.append(port)
            print(f'port {port}: OK')
        else:
            print(f'port {port}: status {r.status_code}')
    except Exception as e:
        print(f'port {port}: FAILED — {e}')

print(f'\\nLive ports: {alive_ports}')"""))

# 26: render_ann_parallel
cells.append(code("""\
_port_cycle = _itools.cycle(alive_ports)
_port_lock  = threading.Lock()
_render_sem = threading.BoundedSemaphore(len(alive_ports))

def render_ann_parallel(ann: dict, max_retries: int = 3) -> Image.Image:
    with _port_lock:
        port = next(_port_cycle)
    url = json_to_url(ann).replace('http://localhost:8081/', f'http://localhost:{port}/')
    with _render_sem:
        for attempt in range(max_retries):
            try:
                from base64 import decodebytes
                response = requests.get(url, timeout=30).content
                return Image.open(io.BytesIO(decodebytes(response))).convert('RGB')
            except Exception:
                if attempt == max_retries - 1: raise
                print(f'[port {port}] crash, restarting...')
                try: _renderer_procs[port].kill()
                except: pass
                env = {**__import__('os').environ, 'PORT': str(port)}
                _renderer_procs[port] = subprocess.Popen(
                    ['node', 'server.js'], cwd=str(RENDERER_DIR), env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(6)

def render_ann(ann): return render_ann_parallel(ann)
print(f'render_ann → parallel across ports {alive_ports}')"""))

# 27: Part swap helpers md
cells.append(md("## 6. Part swap and deletion control helpers"))

# 28: swap helpers
cells.append(code("""\
def swap_part_in_ann(ann: dict, part: str, new_params: dict) -> dict:
    cf = dict(ann)
    cf[f'{part}_model'] = new_params['model']
    if part in PARTS_WITH_COLOR:
        cf[f'{part}_color'] = new_params.get('color', '')
    return cf


def delete_part_in_ann(ann: dict, part: str) -> dict:
    cf = dict(ann)
    cf[f'{part}_model'] = ''
    if part in PARTS_WITH_COLOR:
        cf[f'{part}_color'] = ''
    return cf


def get_concept_dims(part: str, var_old: int, var_new: int):
    c_old = CONCEPT_TO_IDX.get(f'{part}_{var_old}', -1)
    c_new = CONCEPT_TO_IDX.get(f'{part}_{var_new}', -1)
    return c_old, c_new


print('Defined: swap_part_in_ann  delete_part_in_ann  get_concept_dims')"""))

# 29: Z-ordering record md
cells.append(md("""\
## 7. Z-ordering experiment

**The question:** after showing the MCBM an image with species B's part variant,
does the concept bottleneck correctly identify the new part?

`z_raw[donor_dim] > z_raw[src_dim]` should always hold (null = 100%).

GT ceiling: surgically set z_raw[c_donor] = +3, z_raw[c_src] = −3, run label head.
Leakage = P_gt[donor] − P_render[donor] (0 = perfect visual grounding).

**Key MCBM difference:** z values are unbounded real numbers (no sigmoid).
The ordering criterion is the same — positive margin = correct."""))

# 30: z_ordering_record_mcbm
cells.append(code("""\
def z_ordering_record_mcbm(
    ann_orig: dict,
    sid_src: int,
    part: str,
    var_src: int,
    sid_donor: int,
    var_donor: int,
    z_orig: torch.Tensor,      # pre-computed z_raw for this image [26]
    run_through_mcbm,          # closure: PIL -> (z_raw, probs)
    z_to_probs_mcbm,           # closure: z_raw -> probs
) -> dict:
    \"\"\"
    Render source image with donor species' part variant.
    Check: z_cf[c_donor] > z_cf[c_src]?
    GT ceiling: z[c_donor] = MCBM_Z_ACTIVE, z[c_src] = MCBM_Z_INACTIVE.
    \"\"\"
    c_src   = CONCEPT_TO_IDX[f'{part}_{var_src}']
    c_donor = CONCEPT_TO_IDX[f'{part}_{var_donor}']

    ann_cf       = swap_part_in_ann(ann_orig, part, species_part_params[sid_donor][part])
    img_cf       = render_ann(ann_cf)
    z_cf, p_cf   = run_through_mcbm(img_cf)

    z_new  = float(z_cf[c_donor])
    z_old  = float(z_cf[c_src])
    margin = z_new - z_old

    z_gt          = z_orig.clone()
    z_gt[c_donor] = MCBM_Z_ACTIVE
    z_gt[c_src]   = MCBM_Z_INACTIVE
    p_gt          = z_to_probs_mcbm(z_gt)

    row = {
        'sid_src':          sid_src,
        'sid_donor':        sid_donor,
        'part':             part,
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
    }
    if part == 'tail':
        for i in range(PART_VARIANTS['tail']):
            row[f'z_cf_tail_{i}'] = float(z_cf[CONCEPT_TO_IDX[f'tail_{i}']])
    return row

print('Defined: z_ordering_record_mcbm')"""))

# 31: Renderer visual check md
cells.append(md("""\
### 7a. Renderer visual check — per-part swap and deletion grid

Renders a single base image three ways for each body part to verify the renderer
is working correctly. Only the target part should change between columns."""))

# 32: visual check
cells.append(code("""\
if not RENDERER_AVAILABLE:
    print('[skip] Renderer not available.')
else:
    _vsid  = 0
    _v_idx = test_idx_by_species[_vsid][0]
    _v_ann = test_anns[_v_idx]

    _donors = {}
    for _part in PARTS:
        _var_A = species_variant_idx[_vsid][_part]
        for _sid_B in range(N_SPECIES):
            if _sid_B != _vsid and species_variant_idx[_sid_B][_part] != _var_A:
                _donors[_part] = _sid_B
                break

    print(f'Base species: {_vsid} ({spname(_vsid)})')
    print(f'Donor per part: ' + ', '.join(f'{p}→sp{d}' for p, d in _donors.items()))

    _grid = {}
    for _part in PARTS:
        print(f'  Rendering {_part} ...', end=' ', flush=True)
        _donor  = _donors.get(_part)
        _orig   = render_ann(_v_ann)
        _ann_sw = swap_part_in_ann(_v_ann, _part, species_part_params[_donor][_part]) if _donor else None
        _img_sw = render_ann(_ann_sw) if _ann_sw else None
        _ann_dl = delete_part_in_ann(_v_ann, _part)
        _img_dl = render_ann(_ann_dl)
        _grid[_part] = {'orig': _orig, 'swap': (_img_sw, _donor), 'del': _img_dl}
        print('done')

    fig, axes = plt.subplots(len(PARTS), 3, figsize=(10, len(PARTS) * 2.8))
    for ci, ct in enumerate(['Original', 'Swap (part replaced)', 'Deletion (part removed)']):
        axes[0, ci].set_title(ct, fontsize=10, fontweight='bold', pad=6)

    for ri, _part in enumerate(PARTS):
        _g = _grid[_part]
        axes[ri, 0].set_ylabel(_part, fontsize=10, rotation=0,
                               ha='right', labelpad=40, color=PART_COLORS[_part], fontweight='bold')
        axes[ri, 0].imshow(_g['orig']); axes[ri, 0].axis('off')
        _img_sw, _donor = _g['swap']
        if _img_sw:
            axes[ri, 1].imshow(_img_sw)
        axes[ri, 1].axis('off')
        axes[ri, 2].imshow(_g['del']); axes[ri, 2].axis('off')

    plt.suptitle('MCBM renderer visual check: original / swap / deletion', y=1.02)
    plt.tight_layout()
    plt.savefig('fb_mcbm_renderer_smoketest.png', dpi=150, bbox_inches='tight')
    plt.show()"""))

# 33: Define species pairs (once, reused across all gammas)
cells.append(md("## 8. Define species pairs  *(once; reused across all gammas)*"))

cells.append(code("""\
rng = random.Random(42)

all_pairs = {}   # part -> list of (sid_A, var_A, sid_B, var_B)
for part in PARTS:
    pairs = [
        (sid_A, species_variant_idx[sid_A][part],
         sid_B, species_variant_idx[sid_B][part])
        for sid_A, sid_B in combinations(range(N_SPECIES), 2)
        if species_variant_idx[sid_A][part] != species_variant_idx[sid_B][part]
    ]
    if MAX_PAIRS_PER_PART is not None and len(pairs) > MAX_PAIRS_PER_PART:
        pairs = rng.sample(pairs, MAX_PAIRS_PER_PART)
    all_pairs[part] = pairs
    print(f'{part}: {len(pairs)} pairs')

total_renders_per_gamma = sum(len(p) * 2 * MAX_IMGS_PER_SPECIES for p in all_pairs.values())
print(f'\\nTotal renders per gamma:  {total_renders_per_gamma}  '
      f'(~{total_renders_per_gamma*1.5/60:.0f} min at 1.5s each)')
print(f'Total renders for all {len(GAMMAS)} gammas: {total_renders_per_gamma * len(GAMMAS)}')"""))

# 35: Gamma sweep md
cells.append(md("""\
## 9. Gamma sweep — run z-ordering experiment for each γ

Loads MCBM weights + pre-computed avgpool features for each γ,
runs the full z-ordering experiment (all parts, all pairs),
stores results in `results_by_gamma` dict and `combined_df` DataFrame."""))

# 36: Gamma sweep (main loop)
cells.append(code("""\
from concurrent.futures import ThreadPoolExecutor, as_completed

results_by_gamma = {}   # gamma -> list of row dicts

if not RENDERER_AVAILABLE:
    print('[SKIP] Renderer not available.')
else:
    for g in GAMMAS:
        ckpt_path  = mcbm_ckpt(g)
        feats_path = mcbm_feats(g)
        if not ckpt_path.exists():
            print(f'[SKIP] gamma={g}: checkpoint missing')
            continue
        if not feats_path.exists():
            print(f'[SKIP] gamma={g}: features missing')
            continue

        print(f'\\n--- gamma={g} ---')
        W_c, b_c, W_y, b_y, cfg = load_mcbm_weights(ckpt_path)
        backbone_g = load_mcbm_backbone(ckpt_path, device)
        run_fn, z_to_probs_fn = make_mcbm_runner(W_c, b_c, W_y, b_y, backbone_g, device)

        ids_te     = load_split_order(feats_path, 'test')
        avg_te     = load_features(feats_path, 'avgpool', 'test')
        z_te       = compute_z_mcbm(avg_te, W_c, b_c)    # NO sigmoid
        id_to_row  = {int(i): r for r, i in enumerate(ids_te)}
        sids_te    = np.array([int(_meta_idx.loc[int(i), 'species_id']) for i in ids_te])

        # Sanity: species accuracy
        logits_te = z_te @ W_y.T + b_y
        sp_acc    = float((logits_te.argmax(dim=1).numpy() == sids_te).mean())
        print(f'  species accuracy: {sp_acc:.4f}')

        # Build per-species test index lists for this gamma's feature order
        test_idx_by_species_g = {sid: [] for sid in range(N_SPECIES)}
        for local_idx, ann in enumerate(test_anns):
            sid    = int(ann['class_idx'])
            global_id = _FUNNYBIRDS_N_TRAIN + local_idx
            if global_id in id_to_row:
                test_idx_by_species_g[sid].append(local_idx)

        gamma_rows = []

        def _record(args):
            local_idx, sid_ref, part, var_ref, sid_other, var_other, direction = args
            gid = _FUNNYBIRDS_N_TRAIN + local_idx
            row_idx = id_to_row.get(gid)
            if row_idx is None:
                return None
            r = z_ordering_record_mcbm(
                test_anns[local_idx], sid_ref, part, var_ref, sid_other, var_other,
                z_te[row_idx], run_fn, z_to_probs_fn)
            r['direction'] = direction
            r['gamma']     = g
            return r

        for part in PARTS:
            tasks = []
            for sid_A, var_A, sid_B, var_B in all_pairs[part]:
                for local_idx in test_idx_by_species_g[sid_A][:MAX_IMGS_PER_SPECIES]:
                    tasks.append((local_idx, sid_A, part, var_A, sid_B, var_B, 'fwd'))
                for local_idx in test_idx_by_species_g[sid_B][:MAX_IMGS_PER_SPECIES]:
                    tasks.append((local_idx, sid_B, part, var_B, sid_A, var_A, 'bwd'))

            with ThreadPoolExecutor(max_workers=len(alive_ports)) as pool:
                futs = {pool.submit(_record, t): t for t in tasks}
                for f in tqdm(as_completed(futs), total=len(futs), desc=f'gamma={g} {part}'):
                    r = f.result()
                    if r is not None:
                        gamma_rows.append(r)

            _part_df = pd.DataFrame([r for r in gamma_rows if r['part'] == part])
            print(f'  {part}: frac_correct={_part_df["ordering_correct"].mean():.3%}  '
                  f'mean_margin={_part_df["margin"].mean():+.4f}')

        results_by_gamma[g] = gamma_rows
        del backbone_g, avg_te, z_te
        gc.collect()
        torch.cuda.empty_cache()
        print(f'  gamma={g}: {len(gamma_rows)} rows total')

print('\\nDone.')"""))

# 37: Build combined df
cells.append(code("""\
if results_by_gamma:
    combined_df = pd.DataFrame([r for rows in results_by_gamma.values() for r in rows])
    combined_df.to_csv('fb_mcbm_z_ordering_all_gammas.csv', index=False)
    print(f'Saved fb_mcbm_z_ordering_all_gammas.csv  ({len(combined_df)} rows)')
    print(f'Gammas with results: {sorted(combined_df["gamma"].unique())}')
    print(f'Parts: {sorted(combined_df["part"].unique())}')
    display(combined_df.groupby('gamma')['ordering_correct'].mean().rename('frac_correct').to_frame())"""))

# 38: Per-gamma summary table
cells.append(md("## 10. Per-gamma aggregate results"))

cells.append(code("""\
if results_by_gamma:
    summary_rows = []
    for g, rows in results_by_gamma.items():
        df_g = pd.DataFrame(rows)
        for part in PARTS:
            sub = df_g[df_g['part'] == part]
            if sub.empty: continue
            summary_rows.append({
                'gamma':           g,
                'part':            part,
                'n_images':        len(sub),
                'frac_correct':    sub['ordering_correct'].mean(),
                'frac_violations': 1 - sub['ordering_correct'].mean(),
                'mean_margin':     sub['margin'].mean(),
                'std_margin':      sub['margin'].std(),
                'mean_p_cf_donor': sub['p_cf_donor'].mean(),
                'mean_p_gt_donor': sub['p_gt_donor'].mean(),
                'mean_leakage':    (sub['p_gt_donor'] - sub['p_cf_donor']).mean(),
                'sem_leakage':     (sub['p_gt_donor'] - sub['p_cf_donor']).sem(),
            })
    part_summary_all = pd.DataFrame(summary_rows)
    print('Per-gamma × part summary:')
    display(part_summary_all.pivot_table(
        index='part', columns='gamma', values='frac_correct').round(3))"""))

# 40: Headline print
cells.append(code("""\
if results_by_gamma:
    print('=== Z-ORDERING EXPERIMENT SUMMARY (MCBM) ===')
    print(f'{"gamma":>7s}  {"part":6s}  {"frac_correct":>13s}  {"mean_margin":>12s}  n')
    print('-' * 55)
    for _, r in part_summary_all.sort_values(['gamma','part']).iterrows():
        print(f'{r["gamma"]:>7.1f}  {r["part"]:6s}  {r["frac_correct"]:>13.3%}  '
              f'{r["mean_margin"]:>+12.4f}  {int(r["n_images"])}')"""))

# 41: Concept margins per tail variant (tail zoom)
cells.append(md("## 11. Tail: concept margin histograms per variant"))

cells.append(code("""\
if results_by_gamma:
    n_tail = PART_VARIANTS['tail']
    # Show one gamma row of histograms per gamma; compare across gammas per variant
    for g in sorted(results_by_gamma.keys()):
        df_g    = pd.DataFrame(results_by_gamma[g])
        tail_df = df_g[df_g['part'] == 'tail']
        if tail_df.empty: continue

        fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=False)
        axes = axes.flatten()
        for i in range(n_tail):
            ax   = axes[i]
            cidx = CONCEPT_TO_IDX[f'tail_{i}']
            sub  = tail_df[tail_df['c_donor'] == cidx]
            if sub.empty: ax.set_visible(False); continue
            fc = sub['ordering_correct'].mean()
            ax.hist(sub['margin'], bins=30, color=PART_COLORS['tail'], alpha=0.75, edgecolor='white')
            ax.axvline(0, color='red', ls='--', lw=1.5)
            ax.set_title(f'tail_{i}  frac_correct={fc:.2%}', fontsize=9)
            ax.set_xlabel('margin'); ax.set_ylabel('count'); ax.grid(True, alpha=0.3)

        plt.suptitle(f'MCBM γ={g} — Margin distribution per tail concept dim\\n'
                     'Red dashed = 0; left of line = violation', y=1.01, fontsize=11)
        plt.tight_layout()
        plt.savefig(f'fb_mcbm_z_tail_margins_gamma{g}.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved fb_mcbm_z_tail_margins_gamma{g}.png')"""))

# 42: Concept grounding boxplots
cells.append(md("## 12. Concept activations in ORIGINAL images (before swap)"))

cells.append(code("""\
if results_by_gamma:
    n_gammas = len(results_by_gamma)
    fig, axes_all = plt.subplots(2, n_gammas, figsize=(4.5 * n_gammas, 8))
    if n_gammas == 1:
        axes_all = axes_all.reshape(2, 1)

    for gi, g in enumerate(sorted(results_by_gamma.keys())):
        df_g = pd.DataFrame(results_by_gamma[g])
        for ri, (col, title) in enumerate([
            ('z_old_orig', 'z_old_orig (source; should be high)'),
            ('z_new_orig', 'z_new_orig (donor; should be ~0 / low)'),
        ]):
            ax   = axes_all[ri, gi]
            data = [df_g[df_g['part'] == p][col].values for p in PARTS]
            bp   = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black', lw=2))
            for patch, part in zip(bp['boxes'], PARTS):
                patch.set_facecolor(PART_COLORS[part]); patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(PARTS)+1)); ax.set_xticklabels(PARTS, fontsize=8)
            ax.set_title(f'γ={g}  {col}', fontsize=8)
            ax.set_ylabel(col, fontsize=8); ax.grid(True, axis='y', alpha=0.3)
            ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)

    plt.suptitle('MCBM concept activations in ORIGINAL images (no sigmoid — raw z_raw values)',
                 y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig('fb_mcbm_z_grounding_boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved fb_mcbm_z_grounding_boxplots.png')"""))

# 43: Good vs failing tail variants
cells.append(md("## 13. Good vs failing tail variants"))

cells.append(code("""\
if results_by_gamma:
    for g in sorted(results_by_gamma.keys()):
        df_g     = pd.DataFrame(results_by_gamma[g])
        tail_df  = df_g[df_g['part'] == 'tail'].copy()
        if tail_df.empty: continue

        var_fc    = (tail_df.groupby('var_donor')['ordering_correct']
                    .mean().rename('frac_correct').reset_index())
        median_fc = var_fc['frac_correct'].median()
        good_vars = set(var_fc[var_fc['frac_correct'] >= median_fc]['var_donor'])
        bad_vars  = set(var_fc[var_fc['frac_correct'] <  median_fc]['var_donor'])
        tail_df['variant_group'] = tail_df['var_donor'].apply(
            lambda v: 'good' if v in good_vars else 'failing')

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, col, interp in [
            (axes[0], 'z_new_orig',
             'High for failing → species identity in wrong concept dim (backwash)'),
            (axes[1], 'z_old_orig',
             'Low for failing → concept not well grounded in own species'),
        ]:
            good_vals    = tail_df[tail_df['variant_group'] == 'good'][col]
            failing_vals = tail_df[tail_df['variant_group'] == 'failing'][col]
            ax.hist(good_vals,    bins=40, alpha=0.6, color='steelblue', density=True,
                    label=f'good (n={len(good_vals)})')
            ax.hist(failing_vals, bins=40, alpha=0.6, color='crimson',   density=True,
                    label=f'failing (n={len(failing_vals)})')
            ax.set_xlabel(col); ax.set_ylabel('density')
            ax.set_title(f'{col}\\n{interp}', fontsize=8)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.suptitle(f'MCBM γ={g} — Good vs failing tail variants: pre-swap concept activations',
                     y=1.02, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'fb_mcbm_good_vs_failing_tail_gamma{g}.png', dpi=150, bbox_inches='tight')
        plt.show()"""))

# 44: Concept confusion matrix (tail violations)
cells.append(md("## 14. Concept confusion for tail violations"))

cells.append(code("""\
if results_by_gamma:
    n_tail = PART_VARIANTS['tail']
    for g in sorted(results_by_gamma.keys()):
        df_g      = pd.DataFrame(results_by_gamma[g])
        tail_df   = df_g[df_g['part'] == 'tail'].copy()
        tail_viol = tail_df[~tail_df['ordering_correct']]
        if tail_viol.empty:
            print(f'gamma={g}: No tail violations — perfect grounding!'); continue

        z_cf_tail_cols = [f'z_cf_tail_{i}' for i in range(n_tail)]
        missing = [c for c in z_cf_tail_cols if c not in tail_viol.columns]
        if missing:
            print(f'gamma={g}: [WARNING] Missing tail z columns {missing}'); continue

        argmax_dim = tail_viol[z_cf_tail_cols].idxmax(axis=1).str.extract(r'(\\d+)')[0].astype(int)
        tail_viol  = tail_viol.copy()
        tail_viol['argmax_tail_dim'] = argmax_dim.values

        conf = np.zeros((n_tail, n_tail), dtype=int)
        for _, row in tail_viol.iterrows():
            conf[int(row['var_donor']), int(row['argmax_tail_dim'])] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf, cmap='Reds')
        ax.set_xticks(range(n_tail)); ax.set_yticks(range(n_tail))
        ax.set_xticklabels([f'tail_{i}' for i in range(n_tail)], rotation=45, ha='right')
        ax.set_yticklabels([f'tail_{i}' for i in range(n_tail)])
        ax.set_xlabel('argmax z_cf tail dim  (what model activated after swap)')
        ax.set_ylabel('var_donor  (tail variant inserted)')
        ax.set_title(f'MCBM γ={g} — Concept confusion (tail violations, n={len(tail_viol)})\\n'
                     'Row = inserted variant, Col = dim model fired on most')
        for i in range(n_tail):
            for j in range(n_tail):
                if conf[i, j] > 0:
                    ax.text(j, i, str(conf[i, j]), ha='center', va='center', fontsize=8,
                            color='white' if conf[i, j] > conf.max() * 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='count')
        plt.tight_layout()
        plt.savefig(f'fb_mcbm_z_tail_confusion_gamma{g}.png', dpi=150, bbox_inches='tight')
        plt.show()"""))

# 45: Source species breakdown
cells.append(md("## 15. Source species breakdown — violation rate by species"))

cells.append(code("""\
if results_by_gamma:
    for g in sorted(results_by_gamma.keys()):
        df_g    = pd.DataFrame(results_by_gamma[g])
        tail_df = df_g[df_g['part'] == 'tail'].copy()
        if tail_df.empty: continue

        viol_by_src = (tail_df.groupby('sid_src')
                       .agg(n_images=('ordering_correct', 'size'),
                            frac_violations=('ordering_correct', lambda x: 1 - x.mean()))
                       .reset_index()
                       .sort_values('frac_violations', ascending=False))

        med = viol_by_src['frac_violations'].median()
        colors = ['crimson' if v > med else 'steelblue' for v in viol_by_src['frac_violations']]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(viol_by_src['sid_src'].astype(str), viol_by_src['frac_violations'],
               color=colors, alpha=0.8)
        ax.axhline(med, color='black', ls='--', lw=1.5, label=f'median ({med:.2%})')
        ax.set_xlabel('sid_src'); ax.set_ylabel('frac_violations')
        ax.set_title(f'MCBM γ={g} — Tail violation rate by source species  (red = above median)')
        ax.legend(); ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(f'fb_mcbm_viol_by_species_gamma{g}.png', dpi=150, bbox_inches='tight')
        plt.show()"""))

# 46: Downstream probability (margin vs p_cf_donor)
cells.append(md("## 16. Downstream probability effect"))

cells.append(code("""\
if results_by_gamma:
    fig, axes = plt.subplots(1, len(results_by_gamma), figsize=(5 * len(results_by_gamma), 4))
    if len(results_by_gamma) == 1:
        axes = [axes]

    for ax, g in zip(axes, sorted(results_by_gamma.keys())):
        df_g    = pd.DataFrame(results_by_gamma[g])
        tail_df = df_g[df_g['part'] == 'tail']
        if tail_df.empty: continue
        colors  = tail_df['ordering_correct'].map({True: 'steelblue', False: 'crimson'})
        ax.scatter(tail_df['margin'], tail_df['p_cf_donor'], c=colors, s=14, alpha=0.4)
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlabel('margin  (z_new − z_old after swap)')
        ax.set_ylabel('p_cf_donor')
        ax.set_title(f'γ={g}\\nmargin vs P(donor) [tail]')
        ax.grid(True, alpha=0.3)

    axes[-1].legend(handles=[
        Line2D([0],[0],marker='o',color='w',markerfacecolor='steelblue',ms=8,label='correct'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='crimson', ms=8,label='violation'),
    ], fontsize=8)
    plt.suptitle('MCBM: z-ordering margin vs donor species probability (tail)', y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig('fb_mcbm_margin_vs_p_cf.png', dpi=150, bbox_inches='tight')
    plt.show()"""))

# 47: GT ceiling vs render swap
cells.append(code("""\
if results_by_gamma:
    fig, axes = plt.subplots(1, len(results_by_gamma), figsize=(5 * len(results_by_gamma), 4))
    if len(results_by_gamma) == 1:
        axes = [axes]

    for ax, g in zip(axes, sorted(results_by_gamma.keys())):
        df_g    = pd.DataFrame(results_by_gamma[g])
        tail_df = df_g[df_g['part'] == 'tail']
        if tail_df.empty: continue
        colors  = tail_df['ordering_correct'].map({True: 'steelblue', False: 'crimson'})
        ax.scatter(tail_df['p_gt_donor'], tail_df['p_cf_donor'], c=colors, s=14, alpha=0.4)
        _lim = max(tail_df['p_gt_donor'].max(), tail_df['p_cf_donor'].max()) * 1.05
        ax.plot([0, _lim], [0, _lim], 'k--', alpha=0.4, lw=1, label='perfect grounding')
        ax.set_xlabel('p_gt_donor  (GT ceiling)')
        ax.set_ylabel('p_cf_donor  (render swap)')
        ax.set_title(f'γ={g}\\nGT ceiling vs render-swap [tail]', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('MCBM: GT ceiling vs render-swap donor probability (tail)\\n'
                 'Points below diagonal = bottleneck missed visual evidence', y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig('fb_mcbm_p_gt_vs_p_cf.png', dpi=150, bbox_inches='tight')
    plt.show()"""))

# 48: Leakage + concept violation breakdown
cells.append(md("## 17. Leakage score per part + per concept"))

cells.append(code("""\
if results_by_gamma:
    concept_summary_all = []
    for g, rows in results_by_gamma.items():
        df_g = pd.DataFrame(rows)
        for (part, c_donor), sub in df_g.groupby(['part', 'c_donor']):
            concept_summary_all.append({
                'gamma':         g,
                'part':          part,
                'c_donor':       c_donor,
                'concept':       CONCEPT_NAMES[int(c_donor)],
                'n_images':      len(sub),
                'frac_correct':  sub['ordering_correct'].mean(),
                'mean_margin':   sub['margin'].mean(),
            })
    concept_summary_all = pd.DataFrame(concept_summary_all)

    for g in sorted(results_by_gamma.keys()):
        cs = concept_summary_all[concept_summary_all['gamma'] == g]
        ps = part_summary_all[part_summary_all['gamma'] == g]
        if cs.empty or ps.empty: continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        ax = axes[0]
        parts_ordered = ps.sort_values('frac_violations', ascending=False)['part'].tolist()
        ax.bar(parts_ordered,
               ps.set_index('part').loc[parts_ordered, 'mean_leakage'],
               yerr=ps.set_index('part').loc[parts_ordered, 'sem_leakage'],
               color=[PART_COLORS[p] for p in parts_ordered],
               alpha=0.8, capsize=4)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('Mean leakage  (P_gt − P_render)')
        ax.set_title(f'γ={g} — Leakage per body part')
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1]
        cs_sorted = cs.sort_values('frac_correct').head(20)
        ax.barh(cs_sorted['concept'], 1 - cs_sorted['frac_correct'],
                color=[PART_COLORS[CONCEPT_TO_PART[c]] for c in cs_sorted['concept']], alpha=0.8)
        ax.set_xlabel('Violation rate  (1 − frac_correct)')
        ax.set_title(f'γ={g} — Top-20 concept dims by violation rate')
        ax.legend(handles=[Patch(color=c, label=p) for p, c in PART_COLORS.items()], fontsize=7)
        ax.grid(True, axis='x', alpha=0.3)

        plt.suptitle(f'MCBM γ={g}: Leakage and violation breakdown', y=1.02)
        plt.tight_layout()
        plt.savefig(f'fb_mcbm_leakage_breakdown_gamma{g}.png', dpi=150, bbox_inches='tight')
        plt.show()"""))

# 49: CROSS-GAMMA COMPARISON md
cells.append(md("""\
## 18. Cross-gamma comparison

Compare all γ values side by side:
- Fraction of correct z-orderings per part vs γ
- Mean margin per part vs γ
- Mean leakage per part vs γ"""))

# 50: frac_correct vs gamma heatmap
cells.append(code("""\
if results_by_gamma and len(results_by_gamma) > 1:
    pivot_correct = part_summary_all.pivot_table(
        index='part', columns='gamma', values='frac_correct')
    pivot_margin  = part_summary_all.pivot_table(
        index='part', columns='gamma', values='mean_margin')
    pivot_leakage = part_summary_all.pivot_table(
        index='part', columns='gamma', values='mean_leakage')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, piv, title, cmap, center in [
        (axes[0], pivot_correct, 'frac_correct (↑ better)', 'RdYlGn', 0.5),
        (axes[1], pivot_margin,  'mean_margin  (↑ better)', 'RdYlGn', 0.0),
        (axes[2], pivot_leakage, 'mean_leakage (↓ better)', 'RdYlGn_r', 0.0),
    ]:
        import seaborn as sns
        sns.heatmap(piv, ax=ax, annot=True, fmt='.3f', cmap=cmap, center=center,
                    linewidths=0.5, linecolor='white', cbar=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('gamma'); ax.set_ylabel('part')

    plt.suptitle('MCBM gamma sweep — Z-ordering metrics per part × γ', y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig('fb_mcbm_gamma_sweep_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved fb_mcbm_gamma_sweep_heatmap.png')"""))

# 51: frac_correct vs gamma line plot
cells.append(code("""\
if results_by_gamma and len(results_by_gamma) > 1:
    gammas_sorted = sorted(results_by_gamma.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric, ylabel, title in [
        (axes[0], 'frac_correct', 'Fraction correct z-orderings', 'Visual grounding (↑ better)'),
        (axes[1], 'mean_margin',  'Mean margin (z_new − z_old)',  'Ordering confidence (↑ better)'),
        (axes[2], 'mean_leakage', 'Mean leakage (P_gt − P_cf)',   'Leakage (↓ = more grounded)'),
    ]:
        for part in PARTS:
            sub = part_summary_all[part_summary_all['part'] == part].sort_values('gamma')
            ax.plot(sub['gamma'], sub[metric], marker='o', label=part,
                    color=PART_COLORS[part], linewidth=2)
        ax.set_xlabel('γ (IB penalty strength)'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(gammas_sorted); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('MCBM: z-ordering metrics vs γ — does stronger IB improve visual grounding?',
                 y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig('fb_mcbm_gamma_sweep_lineplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved fb_mcbm_gamma_sweep_lineplot.png')"""))

# 52: z_new_orig vs gamma (backwash signature)
cells.append(code("""\
if results_by_gamma and len(results_by_gamma) > 1:
    # z_new_orig: donor concept activation BEFORE swap (pure backwash signature)
    # Higher z_new_orig = model pre-activates concept before seeing the part
    backwash_rows = []
    for g, rows in results_by_gamma.items():
        df_g = pd.DataFrame(rows)
        for part in PARTS:
            sub = df_g[df_g['part'] == part]
            if sub.empty: continue
            backwash_rows.append({
                'gamma': g,
                'part':  part,
                'mean_z_new_orig': sub['z_new_orig'].mean(),
                'std_z_new_orig':  sub['z_new_orig'].std(),
            })
    bw_df = pd.DataFrame(backwash_rows)

    gammas_sorted = sorted(results_by_gamma.keys())
    fig, ax = plt.subplots(figsize=(8, 4))
    for part in PARTS:
        sub = bw_df[bw_df['part'] == part].sort_values('gamma')
        ax.plot(sub['gamma'], sub['mean_z_new_orig'], marker='o', label=part,
                color=PART_COLORS[part], linewidth=2)
        ax.fill_between(sub['gamma'],
                        sub['mean_z_new_orig'] - sub['std_z_new_orig'],
                        sub['mean_z_new_orig'] + sub['std_z_new_orig'],
                        alpha=0.1, color=PART_COLORS[part])
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('γ (IB penalty strength)')
    ax.set_ylabel('mean z_new_orig (raw; no sigmoid)')
    ax.set_title('Backwash signature: donor concept pre-activation vs γ\\n'
                 'Higher = model already "knows" donor concept before seeing the part', fontsize=9)
    ax.set_xticks(gammas_sorted); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fb_mcbm_backwash_vs_gamma.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved fb_mcbm_backwash_vs_gamma.png')"""))

# 53: z_old_orig vs gamma
cells.append(code("""\
if results_by_gamma and len(results_by_gamma) > 1:
    # z_old_orig: source concept activation in original image (should be high)
    # If this drops with gamma: IB is compressing concept representations
    src_act_rows = []
    for g, rows in results_by_gamma.items():
        df_g = pd.DataFrame(rows)
        for part in PARTS:
            sub = df_g[df_g['part'] == part]
            if sub.empty: continue
            src_act_rows.append({
                'gamma': g, 'part': part,
                'mean_z_old_orig': sub['z_old_orig'].mean(),
                'std_z_old_orig':  sub['z_old_orig'].std(),
            })
    sa_df = pd.DataFrame(src_act_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    for part in PARTS:
        sub = sa_df[sa_df['part'] == part].sort_values('gamma')
        ax.plot(sub['gamma'], sub['mean_z_old_orig'], marker='s', label=part,
                color=PART_COLORS[part], linewidth=2, ls='--')
    ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('γ (IB penalty strength)')
    ax.set_ylabel('mean z_old_orig (raw; no sigmoid)')
    ax.set_title('Source concept activation in original image vs γ\\n'
                 'Dropping with γ = IB compresses even correct concept signal', fontsize=9)
    ax.set_xticks(sorted(results_by_gamma.keys())); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fb_mcbm_src_activation_vs_gamma.png', dpi=150, bbox_inches='tight')
    plt.show()"""))

# 54: Per-part violation rate bar (all gammas side by side)
cells.append(code("""\
if results_by_gamma and len(results_by_gamma) > 1:
    gammas_sorted = sorted(results_by_gamma.keys())
    x     = np.arange(len(PARTS))
    width = 0.15
    offsets = np.linspace(-(len(gammas_sorted)-1)/2, (len(gammas_sorted)-1)/2, len(gammas_sorted)) * width

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap_gamma = plt.cm.viridis(np.linspace(0.1, 0.9, len(gammas_sorted)))
    for i, (g, offset, col) in enumerate(zip(gammas_sorted, offsets, cmap_gamma)):
        sub = part_summary_all[part_summary_all['gamma'] == g].set_index('part')
        vals = [1 - sub.loc[p, 'frac_correct'] if p in sub.index else np.nan for p in PARTS]
        ax.bar(x + offset, vals, width, label=f'γ={g}', color=col, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(PARTS)
    ax.set_ylabel('Violation rate  (1 − frac_correct)')
    ax.set_title('MCBM: z-ordering violation rate per part across γ values\\n'
                 '0 = perfect visual grounding', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fb_mcbm_violation_by_part_gamma.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved fb_mcbm_violation_by_part_gamma.png')"""))

# 55: Per-concept violation rate ranked (one curve per gamma)
cells.append(code("""\
if results_by_gamma and len(results_by_gamma) > 1:
    gammas_sorted = sorted(results_by_gamma.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap_gamma = plt.cm.viridis(np.linspace(0.1, 0.9, len(gammas_sorted)))

    for g, col in zip(gammas_sorted, cmap_gamma):
        cs = (concept_summary_all[concept_summary_all['gamma'] == g]
              .sort_values('frac_correct').reset_index(drop=True))
        ax.plot(cs.index, 1 - cs['frac_correct'], marker='o', markersize=4,
                label=f'γ={g}', color=col, linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Concept rank (worst first)')
    ax.set_ylabel('Violation rate  (1 − frac_correct)')
    ax.set_title('MCBM: per-concept violation rate ranked — does γ lift the floor?', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fb_mcbm_concept_violation_ranked.png', dpi=150, bbox_inches='tight')
    plt.show()"""))

# 56: Summary
cells.append(md("## 19. Final summary"))

cells.append(code("""\
if results_by_gamma:
    print('=== MCBM GAMMA SWEEP — FINAL SUMMARY ===')
    print()
    overall = (combined_df.groupby('gamma')['ordering_correct'].mean()
               .rename('frac_correct').reset_index())
    print('Overall frac_correct per gamma:')
    for _, r in overall.iterrows():
        n = len(combined_df[combined_df['gamma'] == r['gamma']])
        print(f'  gamma={r["gamma"]:.1f}: {r["frac_correct"]:.3%}  (n={n})')
    print()
    print('Worst and best part per gamma:')
    for g in sorted(results_by_gamma.keys()):
        sub = part_summary_all[part_summary_all['gamma'] == g].sort_values('frac_violations', ascending=False)
        if sub.empty: continue
        worst = sub.iloc[0]; best = sub.iloc[-1]
        print(f'  gamma={g:.1f}: worst={worst["part"]} ({worst["frac_violations"]:.1%}) '
              f'best={best["part"]} ({best["frac_violations"]:.1%})')
    print()
    print('Interpretation:')
    print('  frac_correct < 100%  → bottleneck not fully visually grounded')
    print('  mean_z_new_orig > 0  → model pre-activates donor concept (backwash)')
    print('  Increasing γ should reduce backwash if IB suppresses species-identity leakage')
else:
    print('No results — run renderer cells first.')"""))

# ── write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": cells,
}

out = Path("funnybird_notebooks/fb_mcbm_renderer_swap_v2.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {out}  ({len(cells)} cells, {out.stat().st_size//1024} KB)")
