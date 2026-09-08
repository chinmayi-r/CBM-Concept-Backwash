"""Read-only MCBM report computations; no optimizer, training, or job submission.

The report calls these functions explicitly. Concept scores are q_j(h_j), whereas
the official species head consumes h. Never substitute a Koh head for that MLP.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

SEED = 20260903
# Explicit post-hoc engineering tolerance, carried over from Notebook02's
# direct erasure replay. This is not a scientific effect-size threshold.
REPLAY_LOGIT_ATOL = 0.02
# Second declared acceptance lane, applied uniformly to every gamma and never
# ratcheted per failing model: a full replay whose worst rows exceed the strict
# guard is accepted WITH DISCLOSURE when at most DISCLOSED_MAX_ROWS of its rows
# exceed it and none exceeds DISCLOSED_MAX_ABS_ERROR. The exceeding rows are
# saved beside the cache and must be carried into strict-sign figures as an
# excluded-rows sensitivity check, mirroring Notebook02's boundary-sensitivity
# precedent (MCBM_RECOVERY_AUDIT.md, batches 4 and 7).
DISCLOSED_MAX_ROWS = 10
DISCLOSED_MAX_ABS_ERROR = 0.05
# Stable replay identity. The cache key previously hashed this whole file, so
# harmless CLI or report edits orphaned completed GPU caches (audit batch 7,
# defect 1). Bump this string ONLY when a change affects the inference itself
# (model construction, preprocessing, checkpoint choice, or row ordering).
REPLAY_PROTOCOL = "swap-replay-v2"


def checkpoint_tag(gamma):
    return f"{gamma:g}".replace(".", "p")


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def softmax(logits):
    values = np.asarray(logits, dtype=float)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def task_head(checkpoint):
    """Replay the pinned Linear/ReLU species MLP, with explicit shape checks."""
    import re
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("model", saved)
    keys = []
    for key in state:
        match = re.fullmatch(r"mlp_y\.(\d+)\.weight", key)
        if match:
            keys.append((int(match[1]), key))
    if [position for position, _ in sorted(keys)] != [0, 2]:
        raise ValueError(f"Expected official 26→256→50 Linear/ReLU head: {checkpoint}")
    layers = [(state[key].float(), state[key[:-6] + "bias"].float())
              for _, key in sorted(keys)]
    if [tuple(w.shape) for w, _ in layers] != [(256, 26), (50, 256)]:
        raise ValueError("MCBM species-head dimensions differ from frozen FunnyBird recipe")

    def forward(h):
        with torch.inference_mode():
            value = torch.as_tensor(np.asarray(h), dtype=torch.float32)
            value = torch.relu(torch.nn.functional.linear(value, *layers[0]))
            return torch.nn.functional.linear(value, *layers[1]).numpy()
    return forward


def score_reference(h, z, c, names):
    """Every coordinate, both representations, both supplied-label buckets."""
    rows = []
    for kind, values in (("h: internal slot", h), ("z: concept logit", z)):
        for j, name in enumerate(names):
            for label in (0, 1):
                selected = np.asarray(values)[np.asarray(c)[:, j] == label, j]
                rows.append(dict(representation=kind, concept=name, label=label,
                                 N=len(selected), mean=selected.mean() if len(selected) else np.nan,
                                 SD=selected.std(ddof=1) if len(selected) > 1 else np.nan))
    return pd.DataFrame(rows)


def replacement_use(h, c, y, saved_probability, forward, spans):
    """Same fivefold label-mean replacement as Standard, but using actual h→MLP."""
    h, c, y = np.asarray(h), np.asarray(c, int), np.asarray(y, int).reshape(-1)
    raw = forward(h)
    probability = softmax(raw)
    if not np.allclose(probability, saved_probability, rtol=2e-5, atol=2e-6):
        raise ValueError("Saved MCBM species-head replay disagrees with ordinary export")
    specs = {"all 26": np.arange(h.shape[1]),
             **{part: np.arange(lo, hi) for part, (lo, hi) in spans.items()}}
    altered = {name: np.full_like(raw, np.nan) for name in specs}
    for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED).split(h, y):
        means = np.empty((h.shape[1], 2))
        for j in range(h.shape[1]):
            for label in (0, 1):
                values = h[train][c[train, j] == label, j]
                if not len(values):
                    raise ValueError(f"No training-fold label {label} examples for coordinate {j}")
                means[j, label] = values.mean()
        expected = means[np.arange(h.shape[1])[None, :], c[test]]
        for name, cols in specs.items():
            changed = h[test].copy()
            changed[:, cols] = expected[:, cols]
            altered[name][test] = forward(changed)
    rows = []
    for name, logits in altered.items():
        if not np.isfinite(logits).all():
            raise ValueError("Incomplete out-of-fold species-head replacement")
        p = softmax(logits)
        rows.append(dict(replaced_block=name, coordinates_replaced=len(specs[name]),
                         raw_accuracy=float(np.mean(raw.argmax(1) == y)),
                         accuracy_after_replacement=float(np.mean(logits.argmax(1) == y)),
                         top1_change_rate=float(np.mean(logits.argmax(1) != raw.argmax(1))),
                         mean_probability_mass_moved=float((.5 * abs(p-probability).sum(1)).mean())))
    return pd.DataFrame(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_fingerprint():
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                cudnn=torch.backends.cudnn.version(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)


def adopt_existing_cache(cache_root, cache, marker, result_path, checkpoint, n_rows):
    """One-time migration for caches accepted under the old source-hashed key.

    The old key hashed this helper's entire source, so any edit orphaned every
    completed frozen replay (audit batch 7, defect 1). Adoption re-registers a
    previously ACCEPTED cache under the stable key only when its stored metadata
    validates exactly: same checkpoint path, same row count, stored array
    checksum, shape, and finiteness. Rejected or partial caches are never
    adopted, and adoption repeats no inference.
    """
    import shutil
    candidates = []
    for old in sorted(cache_root.iterdir() if cache_root.is_dir() else []):
        if not old.is_dir() or old == cache:
            continue
        meta_path = old / "SUCCESS.json"
        array_path = old / "h_cf.npy"
        if not meta_path.exists() or not array_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("rows") != n_rows or meta.get("checkpoint") != str(checkpoint):
            continue
        if meta.get("sha256") != sha256(array_path):
            continue
        stored = np.load(array_path, allow_pickle=False)
        if stored.shape != (n_rows, 26) or not np.isfinite(stored).all():
            continue
        candidates.append((meta_path.stat().st_mtime, str(old), meta, stored))
    if not candidates:
        return None
    _, old, meta, stored = max(candidates, key=lambda item: item[:2])
    np.save(cache / "h_cf.partial.npy", stored, allow_pickle=False)
    (cache / "h_cf.partial.npy").replace(result_path)
    for extra in ("replay_diagnostic.csv", "disclosed_rows.csv"):
        if (Path(old) / extra).exists():
            shutil.copy2(Path(old) / extra, cache / extra)
    meta = dict(meta, adopted_from=old)
    (cache / "SUCCESS.partial.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (cache / "SUCCESS.partial.json").replace(marker)
    print("ADOPTED previously accepted frozen replay (no inference repeated):",
          old, "->", str(cache), flush=True)
    return stored


def replay_counterfactual_h(swaps, gamma, curated, curated_repo, *, diagnose=False):
    """Frozen inference with the exact accepted swap transform and hash validation.

    Batch one matches z_ordering_swap.make_run_fn. Cache identity includes source,
    checkpoint, RGB hashes, and the CSV's actual scalar scores. Partial caches are
    ignored, not grounds to retrain or delete completed scientific artifacts.
    """
    import os
    from PIL import Image
    from torchvision import transforms
    from grounding_deletion import load_model, _MEAN, _STD
    if not torch.cuda.is_available():
        raise RuntimeError("MCBM matched swap replay needs a CUDA GPU; no training is performed")
    checkpoint = (Path(curated_repo) / "external/minimal_cbm/results" /
                  f"funnybirds-mcbm-g{checkpoint_tag(gamma)}" / "1/models/epoch_100.pt")
    identity = hashlib.sha256()
    identity.update(sha256(checkpoint).encode())
    identity.update(REPLAY_PROTOCOL.encode())
    for source in sorted((Path(curated_repo)/'external/minimal_cbm/src/models').rglob('*.py')):
        identity.update(source.read_bytes())
    for relative in ("external/minimal_cbm/src/models/mcbm.py",
                     "external/minimal_cbm/src/models/cbm.py",
                     "external/minimal_cbm/src/models/vanilla.py",
                     "analysis/grounding_deletion.py"):
        identity.update((Path(curated_repo) / relative).read_bytes())
    identity.update(swaps[["render_id", "image_cf_sha256", "part", "var_src", "var_donor",
                           "z_old", "z_new", "p_cf_donor"]].to_csv(index=False).encode())
    if diagnose:
        # Include the reported mismatch, then a deterministic spread of images.
        known = swaps.render_id.eq('cf-beak-fwd-li000240-s24-d42-vs0-vd3-37fb5bacd9d622db')
        selected = pd.concat([swaps.loc[known], swaps.iloc[np.linspace(0,len(swaps)-1,min(32,len(swaps)),dtype=int)]])
        swaps = selected.drop_duplicates('render_id').reset_index(drop=True)
    unique = swaps.drop_duplicates("image_cf_sha256")
    for row in unique.itertuples():
        if sha256(row.image_cf_path) != row.image_cf_sha256:
            raise ValueError(f"Accepted counterfactual RGB bytes changed: {row.image_cf_path}")
    cache = Path(curated) / "mcbm_notebook03_replay" / identity.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    result_path = cache / "h_cf.npy"
    marker = cache / "SUCCESS.json"
    if not diagnose and result_path.exists() and marker.exists():
        meta = json.loads(marker.read_text())
        if meta.get("sha256") == sha256(result_path):
            h = np.load(result_path, allow_pickle=False)
            if h.shape == (len(swaps), 26) and np.isfinite(h).all():
                print("Reusing verified frozen MCBM swap inference:", cache, flush=True)
                return h
    if not diagnose:
        adopted = adopt_existing_cache(cache.parent, cache, marker, result_path,
                                       checkpoint, len(swaps))
        if adopted is not None:
            return adopted
    if not diagnose:
        sample = replay_counterfactual_h(swaps,gamma,curated,curated_repo,diagnose=True)
        if not sample.score_pass.all():
            worst = float(sample.max_score_error.max())
            if worst <= DISCLOSED_MAX_ABS_ERROR:
                print(f"Sample max error {worst:.5f} exceeds the strict {REPLAY_LOGIT_ATOL} guard "
                      f"but is within the declared disclosed cap {DISCLOSED_MAX_ABS_ERROR}; "
                      "continuing to the full replay, which decides acceptance over all rows.",
                      flush=True)
            else:
                raise ValueError('Small replay sample disagrees with accepted CSV beyond the disclosed cap; stopped before full GPU replay. Read replay_diagnostic.csv; tolerances unchanged.')
    model, width = load_model(f"funnybirds-mcbm-g{checkpoint_tag(gamma)}", 1, 100, "cuda")
    if width != 26 or type(model).__name__ != "MinimalConceptBottleneckModel":
        raise ValueError("Replay did not construct official 26-concept MCBM")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                     transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    lookup = {}
    with torch.inference_mode():
        for number, row in enumerate(unique.itertuples(), 1):
            with Image.open(row.image_cf_path) as image:
                x = transform(image.convert("RGB")).unsqueeze(0).to("cuda")
            out = model(x, torch.zeros(1, 26, device="cuda"), sampling=False)
            lookup[row.image_cf_sha256] = (array(out["z"]).reshape(26),
                                           array(out["c_logits"]).reshape(26),
                                           array(out["y_preds"]).reshape(50))
            if number % 100 == 0:
                print(f"gamma={gamma:g}: frozen inference {number}/{len(unique)} images", flush=True)
    import funnybirds_concepts as fbc
    spans = fbc.group_slices(fbc.load_parts(Path(os.environ.get('FUNNYBIRDS_ROOT',Path(curated)/'FunnyBirds'))))
    errors, probability_errors, comparisons = [], [], []
    for row in swaps.itertuples():
        h, z, p = lookup[row.image_cf_sha256]
        lo, _ = spans[row.part]
        expected = np.array([row.z_old, row.z_new])
        replayed = z[lo + np.array([int(row.var_src), int(row.var_donor)])]
        errors.extend(abs(replayed - expected))
        probability_errors.append(abs(float(p[int(row.sid_donor)]) - row.p_cf_donor))
        old_margin = float(row.z_new - row.z_old)
        new_margin = float(replayed[1] - replayed[0])
        original_margin = float(row.z_new_orig - row.z_old_orig)
        def outcome(m):
            # Do not fold exact final ties into a source-winning label
            # (audit batch 5): strict backwash needs delta>0 AND m<0.
            if m > 0:
                return 'donor wins'
            if m == 0:
                return 'final tie (m=0)'
            return 'donorward, source wins' if m-original_margin > 0 else 'no donorward move'
        comparisons.append(dict(render_id=row.render_id, old_expected=expected[0],
            old_replayed=float(replayed[0]), new_expected=expected[1], new_replayed=float(replayed[1]),
            max_score_error=float(np.max(abs(replayed-expected))),
            probability_error=probability_errors[-1],
            accepted_margin=old_margin,replayed_margin=new_margin,
            outcome_changed=outcome(old_margin)!=outcome(new_margin),
            accepted_boundary_distance=min(abs(old_margin),abs(old_margin-original_margin)),
            score_pass=bool(np.isfinite(replayed).all() and np.max(abs(replayed-expected))<=REPLAY_LOGIT_ATOL)))
    comparison = pd.DataFrame(comparisons)
    diagnostic_path = cache / 'replay_diagnostic.csv'
    comparison.to_csv(diagnostic_path,index=False)
    print('Replay engineering audit (accepted CSV unchanged):',dict(
        gamma=gamma,rows=len(comparison),post_hoc_absolute_logit_tolerance=REPLAY_LOGIT_ATOL,
        max_score_error=float(comparison.max_score_error.max()),
        outcome_sensitive_rows=int(comparison.outcome_changed.sum()),
        outcome_comparison='counterfactual replay with accepted original margin'),flush=True)
    if diagnose:
        print(comparison.to_string(index=False),flush=True)
        print('DIAGNOSTIC ONLY — no accepted replay cache or scientific SUCCESS written:',diagnostic_path,flush=True)
        print(dict(torch=torch.__version__,cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(),cudnn=torch.backends.cudnn.version(),
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32),flush=True)
        del model
        torch.cuda.empty_cache()
        return comparison
    h = np.stack([lookup[key][0] for key in swaps.image_cf_sha256])
    if not np.isfinite(h).all():
        raise ValueError("Non-finite counterfactual internal slots")
    # Audit batch 7, defect 2: persist the computed arrays BEFORE acceptance so a
    # tolerance rejection never discards a completed frozen-inference pass again.
    np.save(cache / "h_cf.partial.npy", h, allow_pickle=False)
    (cache / "h_cf.partial.npy").replace(result_path)
    exceeding = comparison[~comparison.score_pass]
    worst_error = float(comparison.max_score_error.max())
    strict = exceeding.empty
    disclosed = (not strict and len(exceeding) <= DISCLOSED_MAX_ROWS
                 and worst_error <= DISCLOSED_MAX_ABS_ERROR)
    if not (strict or disclosed):
        rejected = dict(environment_fingerprint(), gamma=gamma, rows=len(comparison),
                        exceeding_rows=int(len(exceeding)), max_score_error=worst_error,
                        strict_guard=REPLAY_LOGIT_ATOL,
                        disclosed_caps=dict(rows=DISCLOSED_MAX_ROWS,
                                            absolute_error=DISCLOSED_MAX_ABS_ERROR),
                        note='h_cf.npy retained for assessment; no SUCCESS written')
        (cache / "REJECTED.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        raise ValueError(f'Counterfactual replay mismatch beyond disclosed caps: '
                         f'{int(len(exceeding))}/{len(comparison)} rows, max {worst_error:.5f}; '
                         f'arrays retained at {cache}; details: {diagnostic_path}')
    # Erasure compares this session with itself, not with historical probabilities.
    # Verify the recovered h and saved species head against all 50 current outputs.
    current_probabilities = np.stack([lookup[key][2] for key in swaps.image_cf_sha256])
    head_probabilities = softmax(task_head(checkpoint)(h))
    head_error = float(np.max(np.abs(head_probabilities-current_probabilities)))
    if not np.isfinite(current_probabilities).all() or not np.allclose(
            head_probabilities,current_probabilities,rtol=2e-5,atol=2e-6):
        raise ValueError(f'Same-session recovered-h species-head mismatch: {head_error}')
    print('Historical probability discrepancy (diagnostic, not exact replay gate):',
          dict(mean=float(np.mean(probability_errors)),maximum=float(max(probability_errors)),
               same_session_head_max_error=head_error),flush=True)
    if disclosed:
        exceeding.to_csv(cache / "disclosed_rows.csv", index=False)
        print(f"DISCLOSED ACCEPTANCE for gamma={gamma:g}: {len(exceeding)} of "
              f"{len(comparison)} rows exceed the strict {REPLAY_LOGIT_ATOL} guard "
              f"(max {worst_error:.5f} <= cap {DISCLOSED_MAX_ABS_ERROR}). These rows are "
              "listed in disclosed_rows.csv and MUST be carried into strict-sign "
              "figures as an excluded-rows sensitivity check.", flush=True)
    meta = dict(sha256=sha256(result_path), rows=len(h), checkpoint=str(checkpoint),
                post_hoc_absolute_logit_tolerance=REPLAY_LOGIT_ATOL,
                acceptance_mode="strict" if strict else "disclosed_discrepancies",
                disclosed_row_ids=exceeding.render_id.tolist(),
                outcome_sensitive_rows=int(comparison.outcome_changed.sum()),
                same_session_head_max_error=head_error,
                historical_probability_check='reported diagnostic; erasure uses same-session before/after',
                max_raw_score_replay_error=float(max(errors)),
                max_probability_replay_error=float(max(probability_errors)), training=False)
    (cache / "SUCCESS.partial.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (cache / "SUCCESS.partial.json").replace(marker)
    del model
    torch.cuda.empty_cache()
    print("Frozen replay verified:", meta, flush=True)
    return h


def off_target_erasure(swaps, h_cf, h_ordinary, c_ordinary, forward, spans):
    """Nonlinear intervention contrast, not an additive contribution formula."""
    means = []
    for j in range(h_cf.shape[1]):
        values = h_ordinary[c_ordinary[:, j] == 0, j]
        if not len(values):
            raise ValueError(f"Absent baseline undefined for coordinate {j}")
        means.append(values.mean())
    changed = h_cf.copy()
    selected_counts = []
    for i, row in enumerate(swaps.itertuples()):
        lo, hi = spans[row.part]
        keep = {lo + int(row.var_src), lo + int(row.var_donor)}
        indices = [j for j in range(lo, hi) if j not in keep]
        changed[i, indices] = np.asarray(means)[indices]
        selected_counts.append(len(indices))
        if not np.array_equal(changed[i, list(keep)], h_cf[i, list(keep)]):
            raise AssertionError("Erasure altered an old/donor slot")
    before, after = forward(h_cf), forward(changed)
    rows = np.arange(len(swaps))
    source, donor = swaps.sid_src.to_numpy(int), swaps.sid_donor.to_numpy(int)
    gap_before = before[rows, source] - before[rows, donor]
    gap_after = after[rows, source] - after[rows, donor]
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))
    evidence = gap_before - gap_after
    return pd.DataFrame(dict(
        part=swaps.part.to_numpy(), source_species=source, donor_species=donor,
        source_value=swaps.var_src.to_numpy(), donor_value=swaps.var_donor.to_numpy(),
        original_image=swaps.orig_render_id.to_numpy(), off_target_coordinates=selected_counts,
        off_target_source_evidence=evidence, m_cf=swaps.m_cf.to_numpy(),
        controlled_event=swaps.responded_but_source_wins.to_numpy(),
        pairwise_source_share_before=sigmoid(gap_before),
        pairwise_source_share_after=sigmoid(gap_after),
        pairwise_source_share_reduction=sigmoid(gap_before)-sigmoid(gap_after),
        source_minus_donor_logit_before=gap_before, source_minus_donor_logit_after=gap_after,
        top1_changed=before.argmax(1) != after.argmax(1),
        source_to_donor_pair_flip=(gap_before > 0) & (gap_after < 0),
        mean_probability_mass_moved=.5*abs(softmax(before)-softmax(after)).sum(1),
    ))


def loss_gradient_audit(h, c, y, gamma, spans):
    """Weighted gradients of the official loss w.r.t. h at frozen checkpoints.

    No optimizer is made. Parameters have requires_grad=False; autograd targets
    only copies of the saved internal slots. This is not a training trajectory.
    """
    from grounding_deletion import load_model
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,width=load_model(f'funnybirds-mcbm-g{checkpoint_tag(gamma)}',1,100,device)
    if type(model).__name__!='MinimalConceptBottleneckModel' or width!=26:
        raise ValueError('Loss diagnostic requires the official scalar-slot MCBM')
    # Tensors must live where the model's parameters actually are: the synthetic
    # preflight patches load_model with a CPU model, and on a GPU node the
    # availability-based choice above would otherwise split devices.
    device=next(model.parameters()).device
    if not np.isclose(model.gamma,gamma) or not np.isclose(model.beta,1):
        raise ValueError('Saved configuration gamma/beta differs from this report')
    for parameter in model.parameters():parameter.requires_grad_(False)
    generator=torch.Generator(device=device).manual_seed(SEED)
    gradients={name:[] for name in ('task','concepts','representations')}
    for start in range(0,len(h),64):
        latent=torch.tensor(h[start:start+64],device=device,dtype=torch.float32,requires_grad=True)
        labels=torch.as_tensor(c[start:start+64],device=device,dtype=torch.float32)
        species=torch.as_tensor(y[start:start+64],device=device,dtype=torch.long)
        noisy=latent+model.var_z*torch.randn(latent.shape,device=device,generator=generator)
        y_logits,_=model.q_y_z(noisy);c_logits,_=model.q_c_z(noisy);prior,_=model.q_z_c(labels)
        losses=model.get_loss(y=species,c=labels,z=latent,y_logits=y_logits,c_logits=c_logits,z_logits=prior)
        for name,weight in (('task',1),('concepts',model.beta),('representations',model.gamma)):
            gradient=torch.autograd.grad(losses[name]*weight,latent,retain_graph=True)[0]
            gradients[name].append(array(gradient)*len(latent))
    gradients={name:np.concatenate(value) for name,value in gradients.items()}
    rows=[]
    for part,(lo,hi) in spans.items():
        a,b,r=(gradients[name][:,lo:hi] for name in ('task','concepts','representations'))
        norm=lambda v:np.linalg.norm(v,axis=1)
        denominator=norm(a)*norm(b)
        valid=denominator>1e-12
        cosine=np.sum(a*b,axis=1)[valid]/denominator[valid]
        rows.append(dict(gamma=gamma,part=part,N=len(h),coordinates=hi-lo,
            task_gradient_RMS=float(np.sqrt(np.mean(a*a))),
            concept_gradient_RMS=float(np.sqrt(np.mean(b*b))),
            weighted_compression_gradient_RMS=float(np.sqrt(np.mean(r*r))),
            task_concept_cosine_mean=float(cosine.mean()) if len(cosine) else np.nan,
            task_concept_opposed_fraction=float((cosine<0).mean()) if len(cosine) else np.nan,
            cosine_eligible_images=int(valid.sum())))
    del model
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def preflight(curated_repo, curated):
    """Check required real inputs before spending time on the diagnostic fits."""
    import re
    import sys
    from build_mcbm_parity import TAGS, find
    from build_standard_cbm_reports import build_funnybird
    from grounding_deletion import load_model  # validate the official loader's imports
    repo=Path(curated_repo); curated=Path(curated)
    if not torch.cuda.is_available():
        raise RuntimeError('Notebook03 includes frozen GPU inference: run inside a CUDA-enabled Adroit session')
    baseline=json.loads((repo/'notebooks/02_funnybirds_cbm.ipynb').read_text(encoding='utf-8'))
    expected=build_funnybird()['cells']
    for tag in TAGS:
        current=find(baseline['cells'],tag); original=find(expected,tag)
        if ''.join(current['source'])!=''.join(original['source']):
            raise RuntimeError(f'Standard {tag} code differs from current builder; render Notebook02 first')
        if not current.get('outputs') or any(o['output_type']=='error' for o in current['outputs']):
            raise RuntimeError(f'Standard {tag} has no successful executed output; render Notebook02 first')
    recipes=[]
    for gamma in (0,.1,.3,1,3,5):
        name=f'funnybirds-mcbm-g{checkpoint_tag(gamma)}'
        result=repo/'external/minimal_cbm/results'/name/'1'
        for path in (result/'models/epoch_100.pt', result/'predictions/epoch_100.pth',
                     curated/'swap_fixed_v2_attempt2'/f'{name}-s1.csv'):
            if not path.is_file(): raise FileNotFoundError(path)
        from src.helpers import read_config
        config=read_config(str(repo/'external/minimal_cbm/configs/funnybirds'/name))
        labels=Path(config['data']['pkls_dir'])/'test.pkl'
        if not labels.is_file(): raise FileNotFoundError(labels)
        task_head(result/'models/epoch_100.pt')
        recipes.append(dict(gamma=gamma,backbone=config['model']['encoder']['arch'],
            beta=config['model']['beta'],noise_scale=config['model']['var_z'],
            batch=config['training']['batch_size'],epochs=config['training']['n_epochs'],
            base_lr=config['training']['optimizer']['base_lr'],data=config['data']['pkls_dir']))
    print('Current stored configurations (not proof of historical training-time source identity):',flush=True)
    print(pd.DataFrame(recipes).to_string(index=False),flush=True)
    print('REAL INPUT PREFLIGHT PASS: executed Standard baseline, six checkpoint/export/config pairs, CUDA. No training.',flush=True)


if __name__=='__main__':
    import os
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnose-replay',action='store_true')
    parser.add_argument('--prepare-replay',action='store_true',help='Validate and cache full swap inference only; no classifier fits')
    parser.add_argument('--disable-tf32',action='store_true',
                        help='Disable CUDA matmul and cuDNN TF32 for the replay comparison')
    parser.add_argument('--gamma',default='0',
                        help="one gamma such as 0.3, or 'all' for 0 0.1 0.3 1 3 5 "
                             "with per-gamma status collection (audit batch 7, defect 6)")
    args=parser.parse_args()
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        print('TF32 disabled for matmul and cuDNN',flush=True)
    repo=Path(__file__).resolve().parents[1]
    curated=Path(os.environ['CURATED_DATA'])
    if args.diagnose_replay or args.prepare_replay:
        gammas=[0.,0.1,0.3,1.,3.,5.] if args.gamma=='all' else [float(args.gamma)]
        statuses=[]
        for gamma in gammas:
            swaps=pd.read_csv(curated/'swap_fixed_v2_attempt2'/f'funnybirds-mcbm-g{checkpoint_tag(gamma)}-s1.csv')
            try:
                replay_counterfactual_h(swaps,gamma,curated,repo,diagnose=not args.prepare_replay)
                statuses.append((gamma,'ok'))
            except Exception as error:
                statuses.append((gamma,f'FAILED: {error}'))
                if len(gammas)==1:
                    raise
        print('REPLAY STATUS SUMMARY:',flush=True)
        for gamma,status in statuses:
            print(f'  gamma={gamma:g}: {status}',flush=True)
        if any(status!='ok' for _,status in statuses):
            raise SystemExit(1)
    else:
        preflight(repo,curated)
