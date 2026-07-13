#!/usr/bin/env python3
"""Renderer part-swap / z-ordering sweep on the CURATED minimal_cbm models.

Port of ../../run_z_ordering_sweep.py. Renderer + part-swap + species-pairs +
z-ordering record + CSV logic are reused; the three curated changes:
  A. model-loading -> grounding_deletion.load_model (gamma -> config, loop seeds)
  B. concept/species maps -> funnybirds_concepts (no old datasets module)
  C. old "z" (26-d concept logits) -> minimal_cbm out["c_logits"]; z_orig computed
     LIVE (cached per original image); the p_gt path is dropped (minimal_cbm y-head
     reads the bottleneck, not concept logits) -> keep p_cf_donor from out["y_preds"].

For each (species pair, part) it swaps the part to the other species' variant, renders,
and records the concept-logit margin  margin = c_logits[donor] - c_logits[src]  on the
swapped image (grounded => donor variant wins => margin>0). fwd = A's image gets B's
part; bwd = B's image gets A's part. Writes per-part + combined CSVs to <out>/.

Run via train/renderer_swap.slurm (starts the Node renderer). Example:
  python analysis/z_ordering_swap.py --config-prefix funnybirds-mcbm \
     --gammas 0 0.1 0.3 1 3 5 --seeds 1 --funnybirds-root $CURATED_DATA/FunnyBirds \
     --renderer-url http://localhost:8081 --out $CURATED_DATA/swap
"""
from __future__ import annotations
import argparse, gc, io, json, os, random, subprocess, sys, threading, time
from base64 import decodebytes
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
import torchvision.transforms as transforms

try:
    from tqdm import tqdm
except Exception:                       # tqdm optional
    def tqdm(x, **k): return x

HERE = Path(__file__).resolve().parent
CURATED = HERE.parent
for p in (CURATED / "external" / "minimal_cbm", CURATED / "compat", CURATED / "data" / "funnybirds"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

from grounding_deletion import load_model, _MEAN, _STD          # noqa: E402
import funnybirds_concepts as fbc                               # noqa: E402


# ── args ────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--config-prefix", default="funnybirds-mcbm",
                help="funnybirds-mcbm (gamma -> -g<tag>) or funnybirds-cbm (no gamma)")
ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0, 3.0, 5.0])
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--funnybirds-root", required=True)
ap.add_argument("--renderer-url", default="http://localhost:8081")
ap.add_argument("--renderer-dir", default="", help="for auto-restart on death (optional)")
ap.add_argument("--out", required=True)
ap.add_argument("--no-v2", action="store_true", help="skip part-map renders (no pixel_count_cf)")
ap.add_argument("--force", action="store_true")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-pairs", type=int, default=100)
ap.add_argument("--max-imgs", type=int, default=5)
ap.add_argument("--img-size", type=int, default=224)
args = ap.parse_args()

FB = Path(args.funnybirds_root)
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
RENDER_PREFIX = args.renderer_url.rstrip("/") + "/render?"
RENDERER_DIR = Path(args.renderer_dir) if args.renderer_dir else None
USE_V2 = not args.no_v2
FORCE = args.force
N_WORKERS = args.workers
MAX_PAIRS = args.max_pairs
MAX_IMGS = args.max_imgs
MCBM_Z_ACTIVE, MCBM_Z_INACTIVE = 3.0, -3.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  prefix={args.config_prefix}  gammas={args.gammas}  seeds={args.seeds}")

# ── concept / part maps (curated) ───────────────────────────────────────────
parts = fbc.load_parts(FB)
lut = fbc.build_part_lookup(parts)
spans = fbc.group_slices(parts)
CONCEPT_NAMES = fbc.concept_names(parts)
CONCEPT_TO_IDX = {c: i for i, c in enumerate(CONCEPT_NAMES)}
PART_VARIANTS = {p: (b - a) for p, (a, b) in spans.items()}
PARTS = list(parts.keys())
PARTS_WITH_COLOR = {p for p, variants in parts.items() if any("color" in v for v in variants)}
def cidx(part, var): return spans[part][0] + int(var)      # concept index of part's variant

# ── species maps from dataset_test.json ─────────────────────────────────────
test_anns = json.loads((FB / "dataset_test.json").read_text())
N_SPECIES = len({int(a["class_idx"]) for a in test_anns})

def variant_idx_from_ann(ann, part):
    model = ann.get(f"{part}_model", "")
    if not model or model == "placeholder":
        return -1
    kf = {"model": model}
    if part in PARTS_WITH_COLOR:
        col = ann.get(f"{part}_color", "")
        if col:
            kf["color"] = col
    return lut[part].get(tuple(sorted(kf.items())), -1)

species_part_params, species_variant_idx = {}, {}
for ann in test_anns:
    sid = int(ann["class_idx"])
    if sid in species_part_params:
        continue
    species_part_params[sid], species_variant_idx[sid] = {}, {}
    for part in PARTS:
        prm = {"model": ann.get(f"{part}_model", "")}
        if part in PARTS_WITH_COLOR:
            prm["color"] = ann.get(f"{part}_color", "")
        species_part_params[sid][part] = prm
        species_variant_idx[sid][part] = variant_idx_from_ann(ann, part)
test_idx_by_species = {sid: [] for sid in range(N_SPECIES)}
for li, ann in enumerate(test_anns):
    test_idx_by_species[int(ann["class_idx"])].append(li)
print(f"{len(test_anns)} test anns, {N_SPECIES} species")

# ── renderer (verbatim logic, parameterised URL) ────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
PART_SEG_COLORS = {"beak": (255, 255, 0), "eye": (255, 255, 253), "wing": (0, 255, 1),
                   "foot": (255, 0, 1), "tail": (0, 0, 255)}
_restart_lock = threading.Lock()
_server_proc = None

def json_to_url(sample, render_mode="default"):
    url = RENDER_PREFIX + "render_mode=" + render_mode + "&"
    for key in sample:
        if key == "class_idx":
            continue
        url += f"{key}={sample[key]}&"
    return url[:-1]

def json_to_image(sample, part_map=False):
    resp = requests.get(json_to_url(sample, "part_map" if part_map else "default"), timeout=30).content
    img = Image.open(io.BytesIO(decodebytes(resp))).convert("RGB")
    return img.resize((256, 256), Image.NEAREST if part_map else Image.BILINEAR)

def render_ann_safe(ann, max_retries=3):
    global _server_proc
    for attempt in range(max_retries):
        try:
            return json_to_image(ann)
        except Exception:
            if attempt == max_retries - 1:
                raise
            with _restart_lock:
                try:
                    return json_to_image(ann)
                except Exception:
                    pass
                if RENDERER_DIR is not None:
                    try: _server_proc.kill()
                    except Exception: pass
                    _server_proc = subprocess.Popen(
                        ["node", "server.js"], cwd=str(RENDERER_DIR),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(6)

def render_part_map(ann):
    return json_to_image(ann, part_map=True)

def part_pixel_count(img_seg, part):
    arr = np.array(img_seg)
    r, g, b = PART_SEG_COLORS[part]
    return int(((arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)).sum())

def check_renderer_alive(timeout=3.0):
    try:
        return requests.get(json_to_url({
            "beak_model": "beak01.glb", "eye_model": "eye01.glb", "foot_model": "foot01.glb",
            "tail_model": "tail01.glb", "tail_color": "red", "wing_model": "wing01.glb",
            "wing_color": "red", "camera_distance": 300, "camera_pitch": 0, "camera_roll": 0,
            "light_distance": 300, "light_pitch": 0, "light_roll": 0}), timeout=timeout).status_code == 200
    except Exception:
        return False

def swap_part_in_ann(ann, part, new_params):
    cf = dict(ann)
    cf[f"{part}_model"] = new_params["model"]
    if part in PARTS_WITH_COLOR:
        cf[f"{part}_color"] = new_params.get("color", "")
    return cf

def delete_part_in_ann(ann, part):
    cf = dict(ann)
    cf[f"{part}_model"] = ""                # empty model = renderer omits the part
    if part in PARTS_WITH_COLOR:
        cf[f"{part}_color"] = ""
    return cf

def dump_examples(n_per_part=1):
    """Save a few orig/swap/deletion/part_map PNGs so the notebook's inspection grid
    (ref §28/§62) renders WITHOUT a live renderer."""
    exdir = OUT / "examples"; exdir.mkdir(exist_ok=True)
    if any(exdir.glob("*.png")) and not FORCE:
        print(f"[examples] already present -> {exdir}"); return
    for part in PARTS:
        for (a, va, b, vb) in all_pairs[part][:n_per_part]:
            if not test_idx_by_species[a]:
                continue
            base = test_anns[test_idx_by_species[a][0]]
            variants = {"orig": base,
                        "swap": swap_part_in_ann(base, part, species_part_params[b][part]),
                        "delete": delete_part_in_ann(base, part)}
            for tag, ann in variants.items():
                try:
                    render_ann_safe(ann).save(exdir / f"{part}_src{a}_donor{b}_{tag}.png")
                    if USE_V2 and tag == "swap":
                        render_part_map(ann).save(exdir / f"{part}_src{a}_donor{b}_swap_partmap.png")
                except Exception as e:
                    print(f"  [examples] {part} {tag} failed: {e}")
    print(f"[examples] saved -> {exdir}")

# ── inference (curated: c_logits = old "z") ─────────────────────────────────
def make_run_fn(model, n_concepts):
    @torch.inference_mode()
    def run_fn(img_pil):
        x = eval_tf(img_pil).unsqueeze(0).to(device)
        out = model(x, torch.zeros(1, n_concepts, device=device))
        cl = out["c_logits"][0].reshape(-1).float().cpu()      # (n_concepts,) concept logits
        yp = out["y_preds"][0].reshape(-1).float().cpu()       # (n_classes,)
        return cl, yp
    return run_fn

# ── species pairs (fixed seed, matches original) ────────────────────────────
rng = random.Random(42)
all_pairs = {}
for part in PARTS:
    pairs = [(a, species_variant_idx[a][part], b, species_variant_idx[b][part])
             for a, b in combinations(range(N_SPECIES), 2)
             if species_variant_idx[a][part] != species_variant_idx[b][part]
             and species_variant_idx[a][part] >= 0 and species_variant_idx[b][part] >= 0]
    if len(pairs) > MAX_PAIRS:
        pairs = rng.sample(pairs, MAX_PAIRS)
    all_pairs[part] = pairs


def config_for(gamma):
    if "mcbm" in args.config_prefix:
        if gamma == 0:
            tag = "0"
        else:
            tag = str(gamma).replace(".", "p")
            if tag.endswith("p0"):            # 1.0 -> "1", 3.0 -> "3"
                tag = tag[:-2]
        return f"{args.config_prefix}-g{tag}"
    return args.config_prefix


def run_one(config, seed):
    try:
        model, n_concepts = load_model(config, seed, None, device)
    except Exception as e:
        print(f"  [skip] {config} s{seed}: {e}")
        return None
    run_fn = make_run_fn(model, n_concepts)
    combined_csv = OUT / f"{config}-s{seed}.csv"
    if combined_csv.exists() and not FORCE:
        print(f"  [cache] {combined_csv}")
        return pd.read_csv(combined_csv)

    z_orig_cache = {}                                          # (sid, local_idx) -> c_logits
    def z_orig(sid, li):
        k = (sid, li)
        if k not in z_orig_cache:
            cl, _ = run_fn(render_ann_safe(test_anns[li]))
            z_orig_cache[k] = cl
        return z_orig_cache[k]

    all_part_dfs = []
    for part in PARTS:
        part_csv = OUT / f"{config}-s{seed}-{part}.csv"
        if part_csv.exists() and not FORCE:
            all_part_dfs.append(pd.read_csv(part_csv)); continue

        jobs = []
        for a, va, b, vb in all_pairs[part]:
            for li in test_idx_by_species[a][:MAX_IMGS]:
                jobs.append(dict(ann_cf=swap_part_in_ann(test_anns[li], part, species_part_params[b][part]),
                                 sid_src=a, var_src=va, sid_donor=b, var_donor=vb, li=li, direction="fwd"))
            for li in test_idx_by_species[b][:MAX_IMGS]:
                jobs.append(dict(ann_cf=swap_part_in_ann(test_anns[li], part, species_part_params[a][part]),
                                 sid_src=b, var_src=vb, sid_donor=a, var_donor=va, li=li, direction="bwd"))
        if not jobs:
            continue

        # phase 1: threaded renders (I/O)
        renders = [None] * len(jobs)
        def _render(i):
            return i, render_ann_safe(jobs[i]["ann_cf"]), (render_part_map(jobs[i]["ann_cf"]) if USE_V2 else None)
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            for fut in tqdm(as_completed([pool.submit(_render, i) for i in range(len(jobs))]),
                            total=len(jobs), desc=f"  {part} render"):
                i, img_cf, img_seg = fut.result()
                renders[i] = (img_cf, img_seg)

        # phase 2: sequential GPU inference
        rows = []
        for i, job in enumerate(tqdm(jobs, desc=f"  {part} infer")):
            img_cf, img_seg = renders[i]
            c_src, c_donor = cidx(part, job["var_src"]), cidx(part, job["var_donor"])
            cl_cf, p_cf = run_fn(img_cf)
            cl_orig = z_orig(job["sid_src"], job["li"])
            z_new, z_old = float(cl_cf[c_donor]), float(cl_cf[c_src])
            row = dict(sid_src=job["sid_src"], sid_donor=job["sid_donor"], part=part,
                       var_src=job["var_src"], var_donor=job["var_donor"], c_src=c_src, c_donor=c_donor,
                       z_new=z_new, z_old=z_old,
                       z_new_orig=float(cl_orig[c_donor]), z_old_orig=float(cl_orig[c_src]),
                       margin=z_new - z_old, ordering_correct=bool(z_new - z_old > 0),
                       p_cf_donor=float(p_cf[job["sid_donor"]]), direction=job["direction"])
            if USE_V2:
                row["pixel_count_cf"] = part_pixel_count(img_seg, part)
            if part == "tail":
                for ti in range(PART_VARIANTS["tail"]):
                    row[f"z_cf_tail_{ti}"] = float(cl_cf[cidx("tail", ti)])
            rows.append(row)
        del renders; gc.collect()

        part_df = pd.DataFrame(rows)
        part_df.to_csv(part_csv, index=False)
        fwd, bwd = part_df[part_df.direction == "fwd"], part_df[part_df.direction == "bwd"]
        print(f"    {part}: {len(part_df)} rows  fwd_acc={fwd.ordering_correct.mean():.3f} "
              f"bwd_acc={bwd.ordering_correct.mean():.3f}  -> {part_csv.name}")
        all_part_dfs.append(part_df)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not all_part_dfs:
        return None
    out = pd.concat(all_part_dfs, ignore_index=True)
    out.to_csv(combined_csv, index=False)
    print(f"  saved {combined_csv} ({len(out)} rows)")
    return out


def main():
    if not check_renderer_alive():
        print(f"[FATAL] renderer not responding at {args.renderer_url} "
              f"(start it: node server.js in the renderer dir, or use train/renderer_swap.slurm)")
        sys.exit(1)
    dump_examples()
    # CBM has no gamma -> one config; MCBM -> one per gamma. Dedup so a stray --gammas
    # for CBM doesn't re-run the same model.
    is_mcbm = "mcbm" in args.config_prefix
    seen = set()
    for g in (args.gammas if is_mcbm else [0.0]):
        for seed in args.seeds:
            cfg = config_for(g)
            if (cfg, seed) in seen:
                continue
            seen.add((cfg, seed))
            print(f"\n=== {cfg}  seed={seed}{'  (gamma=%s)' % g if is_mcbm else ''} ===")
            run_one(cfg, seed)
    print("\nDone. CSVs in", OUT)


if __name__ == "__main__":
    main()
