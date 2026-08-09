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
import argparse, gc, hashlib, io, json, os, random, subprocess, sys, threading, time
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
ap.add_argument("--koh-checkpoint", default="",
                help="official Koh Joint or X->C checkpoint; bypasses minimal_cbm loading")
ap.add_argument("--koh-class-checkpoint", default="",
                help="official Koh C->Y checkpoint for a two-stage model")
ap.add_argument("--koh-kind", choices=["joint", "two_stage"], default="joint")
ap.add_argument("--koh-name", default="",
                help="output stem required with --koh-checkpoint")
ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.1, 0.3, 1.0, 3.0, 5.0])
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--epoch", type=int, default=None,
                help="evaluate this exact checkpoint epoch for every compared model; "
                     "default loads each model's latest checkpoint")
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
ap.add_argument("--render-cache", default="",
                help="shared directory of fixed rendered PNGs; all compared models must use the same cache")
ap.add_argument("--preflight-only", action="store_true",
                help="run the semantic renderer gate, save its artifacts, and exit before model loading")
ap.add_argument("--renderer-max-reference-mae", type=float, default=15.0,
                help="fail if a live render differs from its stored FunnyBird image by more than this "
                     "mean absolute 8-bit RGB error")
ap.add_argument("--renderer-min-nonblack-frac", type=float, default=0.02,
                help="fail if a live/cached RGB render has less than this fraction of non-black pixels")
ap.add_argument("--renderer-min-rgb-std", type=float, default=5.0,
                help="fail if a live/cached RGB render has lower 8-bit channel standard deviation")
ap.add_argument("--renderer-min-changed-pixels", type=int, default=8,
                help="fail if a preflight swap or deletion changes fewer RGB pixels than this")
ap.add_argument("--renderer-min-part-pixels", type=int, default=4,
                help="fail if a preflight swapped-part map contains fewer target-colour pixels than this")
ap.add_argument("--renderer-max-visibility-attempts", type=int, default=25,
                help="maximum deterministic source-image candidates to inspect per part while "
                     "finding one where both original and swapped parts are visibly present")
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
RENDER_CACHE = Path(args.render_cache) if args.render_cache else None
if RENDER_CACHE is not None:
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)
MCBM_Z_ACTIVE, MCBM_Z_INACTIVE = 3.0, -3.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  prefix={args.config_prefix}  gammas={args.gammas}  "
      f"seeds={args.seeds}  epoch={args.epoch or 'latest'}")
if args.koh_checkpoint and not args.koh_name:
    ap.error("--koh-name is required with --koh-checkpoint")
if args.koh_kind == "two_stage" and args.koh_checkpoint and not args.koh_class_checkpoint:
    ap.error("--koh-class-checkpoint is required for --koh-kind two_stage")

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
_cache_render_lock = threading.Lock()
_server_proc = None

def json_to_url(sample, render_mode="default"):
    url = RENDER_PREFIX + "render_mode=" + render_mode + "&"
    for key in sample:
        if key == "class_idx":
            continue
        url += f"{key}={sample[key]}&"
    return url[:-1]

def json_to_image(sample, part_map=False):
    response = requests.get(
        json_to_url(sample, "part_map" if part_map else "default"), timeout=30)
    response.raise_for_status()
    img = Image.open(io.BytesIO(decodebytes(response.content))).convert("RGB")
    return img.resize((256, 256), Image.NEAREST if part_map else Image.BILINEAR)

def render_ann_safe(ann, part_map=False, max_retries=5):
    """Render with restart/retry for both RGB images and part maps.

    Part maps previously bypassed this function, so one transient disconnect
    killed an hours-long swap job even though ordinary renders were protected.
    """
    global _server_proc
    for attempt in range(max_retries):
        try:
            return json_to_image(ann, part_map=part_map)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            with _restart_lock:
                try:
                    return json_to_image(ann, part_map=part_map)
                except Exception:
                    pass
                if RENDERER_DIR is not None:
                    try: _server_proc.kill()
                    except Exception: pass
                    _server_proc = subprocess.Popen(
                        ["node", "server.js"], cwd=str(RENDERER_DIR),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(6)
                else:
                    time.sleep(min(2 ** attempt, 8))
            print(f"  [renderer retry {attempt + 1}/{max_retries - 1}] "
                  f"{'part_map' if part_map else 'image'}: {exc}")

def render_part_map(ann):
    return render_ann_safe(ann, part_map=True)

def _ann_digest(ann):
    payload = json.dumps(ann, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]

def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB").copy()

def _load_or_repair_cached_png(path, render_fn):
    """Load once on the fast path; atomically replace a missing/corrupt entry."""
    try:
        return _open_rgb(path)
    except (OSError, ValueError):
        pass

    with _cache_render_lock:
        # Another worker may have repaired the entry while this worker waited.
        try:
            return _open_rgb(path)
        except (OSError, ValueError):
            _atomic_save_png(render_fn(), path)
            return _open_rgb(path)

def _atomic_save_png(img, path):
    """Publish a complete PNG in one rename so killed jobs cannot poison cache."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        img.save(tmp, format="PNG")
        # Force a complete decode before the file becomes visible as a cache hit.
        _open_rgb(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

def _rgb_stats(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    nonblack = np.any(arr > 5, axis=2)
    return {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "nonblack_pixels": int(nonblack.sum()),
        "nonblack_fraction": float(nonblack.mean()),
        "rgb_std": float(arr.astype(np.float32).std()),
        "unique_rgb": int(np.unique(arr.reshape(-1, 3), axis=0).shape[0]),
    }

def _assert_rgb_nondegenerate(img, context):
    stats = _rgb_stats(img)
    failures = []
    if stats["nonblack_fraction"] < args.renderer_min_nonblack_frac:
        failures.append(
            f"nonblack_fraction={stats['nonblack_fraction']:.6f} "
            f"< {args.renderer_min_nonblack_frac}"
        )
    if stats["rgb_std"] < args.renderer_min_rgb_std:
        failures.append(f"rgb_std={stats['rgb_std']:.3f} < {args.renderer_min_rgb_std}")
    if stats["unique_rgb"] < 16:
        failures.append(f"unique_rgb={stats['unique_rgb']} < 16")
    if failures:
        raise RuntimeError(
            f"degenerate renderer RGB for {context}: " + "; ".join(failures)
        )
    return stats

def _changed_pixels(img_a, img_b):
    a = np.asarray(img_a.convert("RGB"), dtype=np.int16)
    b = np.asarray(img_b.convert("RGB"), dtype=np.int16)
    return int((np.max(np.abs(a - b), axis=2) > 2).sum())

def _stored_test_image(li, ann):
    return FB / "test" / str(int(ann["class_idx"])) / f"{li:06d}.png"

def render_cached_pair(ann, render_id, need_part_map):
    """Return a fixed RGB render and its cached part map.

    Rendering is serialized because the Node renderer uses shared scene state.
    Once written, every model loads identical PNG bytes from this cache.
    """
    if RENDER_CACHE is None:
        rgb = render_ann_safe(ann)
        seg = render_part_map(ann) if need_part_map else None
        return rgb, seg, "", "", ""

    rgb_path = RENDER_CACHE / "rgb" / f"{render_id}.png"
    seg_path = RENDER_CACHE / "part_map" / f"{render_id}.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    if need_part_map:
        seg_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = _load_or_repair_cached_png(rgb_path, lambda: render_ann_safe(ann))
    seg = (
        _load_or_repair_cached_png(seg_path, lambda: render_part_map(ann))
        if need_part_map else None
    )
    # Do not let an old poisoned cache bypass the live-renderer preflight.
    _assert_rgb_nondegenerate(rgb, f"cache render_id={render_id} path={rgb_path}")
    rgb_sha = _file_sha256(rgb_path)
    seg_sha = _file_sha256(seg_path) if need_part_map else ""
    return rgb, seg, str(rgb_path), rgb_sha, seg_sha

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

def check_renderer_semantic_validity():
    """Fail closed unless this renderer produces real, faithful interventions.

    Determinism alone is insufficient: the July 29 renderer returned the same
    nearly-black PNG for every request and therefore passed the old check.
    """
    ann = test_anns[0]
    rgb_a_img = render_ann_safe(ann)
    rgb_b_img = render_ann_safe(ann)
    seg_a_img = render_part_map(ann)
    seg_b_img = render_part_map(ann)
    rgb_a = np.asarray(rgb_a_img)
    rgb_b = np.asarray(rgb_b_img)
    seg_a = np.asarray(seg_a_img)
    seg_b = np.asarray(seg_b_img)
    if not np.array_equal(rgb_a, rgb_b) or not np.array_equal(seg_a, seg_b):
        raise RuntimeError(
            "renderer is not deterministic for repeated sequential requests; "
            "fixed RGB caching remains possible, but its separately rendered part map "
            "cannot be assumed to describe the RGB geometry"
        )
    canonical_stats = _assert_rgb_nondegenerate(rgb_a_img, "repeated canonical live render")

    reference_path = _stored_test_image(0, ann)
    if not reference_path.exists():
        raise RuntimeError(
            f"renderer semantic preflight requires stored reference image {reference_path}; "
            "an HTTP 200 response is not sufficient"
        )
    reference = _open_rgb(reference_path).resize((256, 256), Image.BILINEAR)
    reference_mae = float(np.abs(
        np.asarray(reference, dtype=np.float32) -
        np.asarray(rgb_a_img, dtype=np.float32)
    ).mean())
    if reference_mae > args.renderer_max_reference_mae:
        raise RuntimeError(
            f"live renderer does not reproduce stored FunnyBird reference: "
            f"pixel_mae={reference_mae:.3f} > {args.renderer_max_reference_mae}; "
            f"reference={reference_path}"
        )

    audit_dir = OUT / "renderer_preflight"
    audit_dir.mkdir(parents=True, exist_ok=True)
    reference.save(audit_dir / "canonical_stored.png")
    rgb_a_img.save(audit_dir / "canonical_live.png")

    report = {
        "canonical_local_index": 0,
        "canonical_reference": str(reference_path),
        "canonical_reference_mae": reference_mae,
        "canonical_live_stats": canonical_stats,
        "parts": [],
    }
    panels = []
    failures = []
    for part in PARTS:
        if not all_pairs.get(part):
            failures.append(f"{part}: no differing-variant pair")
            continue

        # A present 3-D part need not be visible from every camera angle. The old
        # gate blindly used the first source image, so a fully occluded original
        # part made a valid deletion look like a renderer failure. Deterministically
        # choose the first candidate whose original AND swapped part maps show the
        # target; only then test whether swap/deletion changed RGB.
        selected = None
        attempts = 0
        last_visibility = None
        for sid_src, _, sid_donor, _ in all_pairs[part]:
            for li in test_idx_by_species[sid_src]:
                attempts += 1
                orig_ann = test_anns[li]
                swap_ann = swap_part_in_ann(
                    orig_ann, part, species_part_params[sid_donor][part])
                orig = render_ann_safe(orig_ann)
                orig_seg = render_part_map(orig_ann)
                swap = render_ann_safe(swap_ann)
                swap_seg = render_part_map(swap_ann)
                orig_target_pixels = part_pixel_count(orig_seg, part)
                swap_target_pixels = part_pixel_count(swap_seg, part)
                last_visibility = {
                    "local_index": li,
                    "sid_src": sid_src,
                    "sid_donor": sid_donor,
                    "orig_part_pixels": orig_target_pixels,
                    "swap_part_pixels": swap_target_pixels,
                }
                if (orig_target_pixels >= args.renderer_min_part_pixels and
                        swap_target_pixels >= args.renderer_min_part_pixels):
                    delete_ann = delete_part_in_ann(orig_ann, part)
                    delete = render_ann_safe(delete_ann)
                    selected = (
                        sid_src, sid_donor, li, orig, swap, delete,
                        orig_seg, swap_seg, orig_target_pixels, swap_target_pixels,
                    )
                    break
                if attempts >= args.renderer_max_visibility_attempts:
                    break
            if selected is not None or attempts >= args.renderer_max_visibility_attempts:
                break

        if selected is None:
            failures.append(
                f"{part}: no candidate with visible original and swapped target "
                f"within {attempts} attempts; last={last_visibility}"
            )
            report["parts"].append({
                "part": part,
                "status": "failed_visibility_selection",
                "attempts": attempts,
                "last_visibility": last_visibility,
            })
            continue

        (sid_src, sid_donor, li, orig, swap, delete, orig_seg, swap_seg,
         orig_target_pixels, swap_target_pixels) = selected
        # Save evidence before evaluating it. A failed gate must remain visually
        # diagnosable instead of leaving only canonical images and an exception.
        stem = f"{part}_src{sid_src}_donor{sid_donor}"
        orig.save(audit_dir / f"{stem}_orig.png")
        swap.save(audit_dir / f"{stem}_swap.png")
        delete.save(audit_dir / f"{stem}_delete.png")
        orig_seg.save(audit_dir / f"{stem}_orig_partmap.png")
        swap_seg.save(audit_dir / f"{stem}_swap_partmap.png")

        failure = None
        try:
            orig_stats = _assert_rgb_nondegenerate(orig, f"{part} preflight original")
            _assert_rgb_nondegenerate(swap, f"{part} preflight swap")
            _assert_rgb_nondegenerate(delete, f"{part} preflight deletion")
        except RuntimeError as exc:
            orig_stats = None
            failure = str(exc)
        changed_swap = _changed_pixels(orig, swap)
        changed_delete = _changed_pixels(orig, delete)
        if failure is None and changed_swap < args.renderer_min_changed_pixels:
            failure = (
                f"renderer swap did not visibly change {part}: "
                f"changed_pixels={changed_swap} < {args.renderer_min_changed_pixels}"
            )
        if failure is None and changed_delete < args.renderer_min_changed_pixels:
            failure = (
                f"renderer deletion did not visibly change {part}: "
                f"changed_pixels={changed_delete} < {args.renderer_min_changed_pixels}"
            )
        if failure is not None:
            failures.append(f"{part}: {failure}")

        panels.append((part, orig, swap, delete, orig_seg, swap_seg))
        part_report = {
            "part": part,
            "status": "failed" if failure else "passed",
            "failure": failure,
            "visibility_attempts": attempts,
            "local_index": li,
            "sid_src": sid_src,
            "sid_donor": sid_donor,
            "orig_stats": orig_stats,
            "swap_changed_pixels": changed_swap,
            "delete_changed_pixels": changed_delete,
            "orig_part_pixels": orig_target_pixels,
            "swap_part_pixels": swap_target_pixels,
        }
        report["parts"].append(part_report)

    # One durable artifact makes literal visual inspection possible after the job.
    from PIL import ImageDraw
    cell_w, cell_h, label_h = 256, 256, 20
    sheet = Image.new(
        "RGB", (5 * cell_w, max(1, len(panels)) * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (part, orig, swap, delete, orig_seg, swap_seg) in enumerate(panels):
        y = row * (cell_h + label_h)
        for col, (tag, img) in enumerate(
                (("orig", orig), ("swap", swap), ("delete", delete),
                 ("orig_part_map", orig_seg), ("swap_part_map", swap_seg))):
            x = col * cell_w
            draw.text((x + 4, y + 3), f"{part} {tag}", fill="black")
            sheet.paste(img, (x, y + label_h))
    sheet.save(audit_dir / "renderer_semantic_preflight.png")
    report["failures"] = failures
    (audit_dir / "renderer_semantic_preflight.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if failures:
        raise RuntimeError(
            "renderer semantic preflight failed after saving diagnostic artifacts: " +
            " | ".join(failures)
        )
    print(
        "[renderer semantic preflight PASS] deterministic; "
        f"reference_mae={reference_mae:.3f}; every part swap/delete changed RGB; "
        f"part maps contain target pixels -> {audit_dir}"
    )

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
    # Always regenerate these from the renderer that passed this job's semantic
    # preflight. Reusing an older examples directory hid the July 29 corruption.
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
    print(f"[examples] regenerated from validated live renderer -> {exdir}")

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


def _torch_load_model(path):
    """Load a full-model Koh checkpoint across old/new torch defaults."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch before weights_only existed
        return torch.load(path, map_location=device)


def load_koh_run_fn():
    koh = CURATED / "external" / "ConceptBottleneck"
    if str(koh) not in sys.path:
        sys.path.insert(0, str(koh))
    model = _torch_load_model(args.koh_checkpoint).to(device).eval()
    class_model = None
    if args.koh_class_checkpoint:
        class_model = _torch_load_model(args.koh_class_checkpoint).to(device).eval()
    koh_tf = transforms.Compose([
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2.0, 2.0, 2.0]),
    ])

    @torch.inference_mode()
    def run_fn(img_pil):
        x = koh_tf(img_pil).unsqueeze(0).to(device)
        outputs = model(x)
        if args.koh_kind == "joint":
            y_logits = outputs[0]
            concept_outputs = outputs[1:]
        else:
            concept_outputs = outputs
            concept_logits = torch.cat(
                [value.reshape(value.shape[0], -1) for value in concept_outputs], dim=1)
            y_logits = class_model(torch.sigmoid(concept_logits))
        concept_logits = torch.cat(
            [value.reshape(value.shape[0], -1) for value in concept_outputs], dim=1)
        if concept_logits.shape[1] != len(CONCEPT_NAMES):
            raise RuntimeError(
                f"Koh checkpoint emitted {concept_logits.shape[1]} concepts; "
                f"expected {len(CONCEPT_NAMES)}"
            )
        return concept_logits[0].float().cpu(), torch.softmax(y_logits[0], dim=0).float().cpu()

    print(f"[grounding] loaded official Koh checkpoint {args.koh_checkpoint} ({args.koh_kind})")
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
    if args.koh_checkpoint:
        run_fn = load_koh_run_fn()
    else:
        try:
            model, n_concepts = load_model(config, seed, args.epoch, device)
        except Exception as e:
            print(f"  [skip] {config} s{seed}: {e}")
            return None
        run_fn = make_run_fn(model, n_concepts)
    combined_csv = OUT / f"{config}-s{seed}.csv"
    if combined_csv.exists() and not FORCE:
        print(f"  [cache] {combined_csv}")
        return pd.read_csv(combined_csv)

    z_orig_cache = {}                                          # (sid, local_idx) -> (c_logits, metadata)
    def z_orig(sid, li):
        k = (sid, li)
        if k not in z_orig_cache:
            ann = test_anns[li]
            rid = f"orig-li{li:06d}-{_ann_digest(ann)}"
            img, _, path, sha, _ = render_cached_pair(ann, rid, False)
            cl, _ = run_fn(img)
            z_orig_cache[k] = (cl, rid, path, sha)
        return z_orig_cache[k]

    all_part_dfs = []
    for part in PARTS:
        part_csv = OUT / f"{config}-s{seed}-{part}.csv"
        if part_csv.exists() and not FORCE:
            all_part_dfs.append(pd.read_csv(part_csv)); continue

        jobs = []
        for a, va, b, vb in all_pairs[part]:
            for li in test_idx_by_species[a][:MAX_IMGS]:
                ann_cf = swap_part_in_ann(test_anns[li], part, species_part_params[b][part])
                jobs.append(dict(ann_cf=ann_cf, sid_src=a, var_src=va, sid_donor=b,
                                 var_donor=vb, li=li, direction="fwd"))
            for li in test_idx_by_species[b][:MAX_IMGS]:
                ann_cf = swap_part_in_ann(test_anns[li], part, species_part_params[a][part])
                jobs.append(dict(ann_cf=ann_cf, sid_src=b, var_src=vb, sid_donor=a,
                                 var_donor=va, li=li, direction="bwd"))
        if not jobs:
            continue

        # phase 1: threaded renders (I/O)
        renders = [None] * len(jobs)
        def _render(i):
            job = jobs[i]
            rid = (f"cf-{part}-{job['direction']}-li{job['li']:06d}-"
                   f"s{job['sid_src']:02d}-d{job['sid_donor']:02d}-"
                   f"vs{job['var_src']}-vd{job['var_donor']}-{_ann_digest(job['ann_cf'])}")
            rgb, seg, path, sha, seg_sha = render_cached_pair(job["ann_cf"], rid, USE_V2)
            return i, rgb, seg, rid, path, sha, seg_sha
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            for fut in tqdm(as_completed([pool.submit(_render, i) for i in range(len(jobs))]),
                            total=len(jobs), desc=f"  {part} render"):
                i, img_cf, img_seg, rid, path, sha, seg_sha = fut.result()
                renders[i] = (img_cf, img_seg, rid, path, sha, seg_sha)

        # phase 2: sequential GPU inference
        rows = []
        for i, job in enumerate(tqdm(jobs, desc=f"  {part} infer")):
            img_cf, img_seg, rid, image_path, image_sha, partmap_sha = renders[i]
            c_src, c_donor = cidx(part, job["var_src"]), cidx(part, job["var_donor"])
            cl_cf, p_cf = run_fn(img_cf)
            cl_orig, orig_rid, orig_path, orig_sha = z_orig(job["sid_src"], job["li"])
            z_new, z_old = float(cl_cf[c_donor]), float(cl_cf[c_src])
            z_new_orig = float(cl_orig[c_donor])
            z_old_orig = float(cl_orig[c_src])
            margin = z_new - z_old
            margin_orig = z_new_orig - z_old_orig
            response_delta = margin - margin_orig
            row = dict(sid_src=job["sid_src"], sid_donor=job["sid_donor"], part=part,
                       var_src=job["var_src"], var_donor=job["var_donor"], c_src=c_src, c_donor=c_donor,
                       li=job["li"], render_id=rid, image_cf_path=image_path,
                       image_cf_sha256=image_sha, partmap_cf_sha256=partmap_sha,
                       orig_render_id=orig_rid, image_orig_path=orig_path,
                       image_orig_sha256=orig_sha,
                       z_new=z_new, z_old=z_old,
                       z_new_orig=z_new_orig, z_old_orig=z_old_orig,
                       margin=margin, margin_orig=margin_orig,
                       response_delta=response_delta,
                       swap_moved_toward_donor=bool(response_delta > 0),
                       ordering_correct=bool(margin > 0),
                       p_cf_donor=float(p_cf[job["sid_donor"]]), direction=job["direction"])
            if USE_V2:
                row["pixel_count_cf"] = part_pixel_count(img_seg, part)
            # per-variant activations for THIS part -> enables the confusion matrix for
            # every part (not just tail), so the "collapses onto one concept" claim can be
            # checked against grounded parts as a control.
            for ti in range(PART_VARIANTS[part]):
                row[f"z_cf_{part}_{ti}"] = float(cl_cf[cidx(part, ti)])
            rows.append(row)
        del renders; gc.collect()

        part_df = pd.DataFrame(rows)
        part_df.to_csv(part_csv, index=False)
        fwd, bwd = part_df[part_df.direction == "fwd"], part_df[part_df.direction == "bwd"]
        print(f"    {part}: {len(part_df)} rows  fwd_acc={fwd.ordering_correct.mean():.3f} "
              f"bwd_acc={bwd.ordering_correct.mean():.3f}  -> {part_csv.name}")
        all_part_dfs.append(part_df)

    # `model` exists only on the legacy minimal-CBM path.  The Koh model is
    # captured by `run_fn`, so deleting an undefined local here made an otherwise
    # completed Koh sweep stop before writing its combined CSV.
    del run_fn
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
    check_renderer_semantic_validity()
    dump_examples()
    if args.preflight_only:
        print("[preflight-only] renderer gate passed; no model was loaded and no swap CSV was written")
        return
    # CBM has no gamma -> one config; MCBM -> one per gamma. Dedup so a stray --gammas
    # for CBM doesn't re-run the same model.
    if args.koh_checkpoint:
        run_one(args.koh_name, args.seeds[0])
        print("\nDone. CSVs in", OUT)
        return
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
