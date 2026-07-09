# Curated pipeline — decisions, deviations, problems (living doc)

The lab notebook for the `curated/` restart. Every non-obvious choice, every
deviation from the official repos, and every bug found lives here, with the
rationale phrased so it can drop straight into the paper's Methods / Limitations.
Keep it updated as things change. Pinned upstream commits: `external/COMMITS.txt`.

Status legend: ✅ done · ⚠️ partial / caveat · ❌ open · 🔬 verified-in-code

---

## A. Deviations from the official repositories (Methods / reproducibility)

No file inside `external/` is edited — the submodules stay verbatim and citable
at their pinned SHAs. All adaptation lives in `curated/` (compat shims + configs).

| Upstream | What we do differently | Why | Where |
|---|---|---|---|
| minimal_cbm | Add a **FunnyBirds loader** by monkeypatching `src.datasets.get_loader` at runtime | Repo only ships CUB200/CIFAR10/disentanglement/CelebA/Spirals; FunnyBirds isn't registered | `compat/mcbm_funnybirds.py`, `train/run_mcbm.py` |
| minimal_cbm | Force **wandb offline** (`TrainExperiment.wandb_offline=True` + env) | Repo hardcodes an author's wandb key and inits **online**; on a cluster that fails or leaks to a stranger's account | `train/run_mcbm.py` |
| minimal_cbm | Author our own **configs to the real schema**, generated per-run from templates | Bundled examples are `configs/cub12/*`; our datasets/backbones need their own | `train/configs/*`, `train/_paths.sh` |
| minimal_cbm | **Val-based model selection** via a `<pkls>_trainval` dir | Repo evaluates on `test` every epoch (no val); selecting on that leaks test | `data/make_val_split.py`, `train/_paths.sh` |
| ConceptBottleneck (yewsiang) | Kept as a **separate, labeled** official-CBM cross-check, not mixed into the same-backbone comparison | Different codebase/backbone; only for the canonical 3-regime CUB reference | `train/cbm_cub.sh` |

Generated configs/results are written **inside** `external/minimal_cbm/{configs,results}/`
(the trainer keys off those paths). This dirties the submodule *working tree* but
touches no tracked file. ⚠️ open cleanup: redirect results outside `external/`.

---

## B. Facts verified directly in the official code (so results are interpretable)

- 🔬 **Minimality force = `gamma × 0.2 × MSE(z, ±3)`**, target `z = 6c − 3`
  (`src/models/mcbm.py: get_loss_z`, `get_loss`, `q_z_c`). The `0.2` is baked in.
  ⇒ the old hand-written training under-scaled gamma; the sweep goes to `gamma=30`
  (effective 6.0) to reach the strong-bottleneck regime the old runs never did.
- 🔬 **CBM ≠ MCBM at gamma=0.** `MCBM.forward` injects `var_z` noise into z during
  training (`sampled_z = z + var_z*randn`, `sampling=True` in the train loop);
  `CBM.forward` uses z directly. So gamma=0 MCBM is "MCBM architecture, no
  minimality, still stochastic z" — a valid ablation, **not** the CBM reference.
  The true reference is `model_type: cbm` (`train/configs/*-cbm.yaml`).

---

## C. Methodological decisions (rationale for the paper)

1. **Base-case-first ordering.** vanilla → CBM (fully analyzed) → MCBM gamma sweep.
   The instruments (data analysis, recall, swap, grounding) are validated on the
   reference before the treatment. `train/run_baselines.sh` then
   `train/mcbm_gamma_sweep.sh`.
2. **Same-backbone comparability.** vanilla/CBM/MCBM run through one trainer with
   one encoder, so the only variable is head + gamma. **Primary backbone: resnet50,
   held FIXED.** Optional robustness axis is **capacity (resnet18 vs resnet50)** —
   *not* family: resnet50 ≈ inception_v3 in size, so that pairing would test
   nothing. One knob: `ARCH=`.
3. **FunnyBirds vs CUB division of labor.**
   - FunnyBirds concepts are **species-constant** (verified: within-species std ≈ 0
     on test; notebook 01 §2b). So the matched-pair **recall gap is confounded with
     species and n=10 underpowered** → appendix only. The **causal concept-swap**
     is the FunnyBirds headline: it renders/intervenes the *same species* with a
     *different part variant*, severing the species↔concept correlation, so whether
     the prediction follows the part (grounded) or the species (backwash) is
     identifiable.
   - CUB attributes **vary within species** → correlation broken naturally → CUB
     carries the correlational **recall-gap-vs-gamma** axis.
4. **Probe-grounding confound → resolved by intervention, not a better probe.** At
   layer4 species is ~99% linearly decodable, so a single-layer concept probe can
   succeed by decoding species and table-lookup rather than detecting the part.
   Correlational methods can't escape this on a species-constant dataset; the
   counterfactual **swap** does (see C.3). Optional secondary diagnostic: concept
   emergence *across layers* relative to species.
5. **Model selection = validation-based (standard, Koh et al. 2020).** Hold out a
   val fold, select the checkpoint on val, touch test **once**. minimal_cbm has no
   val, so `data/make_val_split.py` builds `<pkls>_trainval/` where the trainer's
   `test.pkl` **is the val fold**; the real test stays untouched for final eval.
   `_paths.sh` auto-prefers `_trainval` and **warns** if it's absent (else per-epoch
   eval is on test = leak). FunnyBirds: carve 10%/class from 50k train. CUB: reuse
   the official `val.pkl`.
6. **Concept-vector gate.** `data/funnybirds/validate_concept_vectors.py` checks our
   26-slot vectors match the official parts.json indexing **before** training.
   Passed: 0 / 50,500 mismatches.
7. **Absent part = all-zero concept group** (~1/4 of images per part). Treated as a
   legitimate state; documented so reviewers know it's intentional.

---

## D. Bugs found and fixed during curation

| Bug | Impact if shipped | Fix |
|---|---|---|
| sweep `sed` replaced **every** `gamma:` line | clobbered `scheduler.gamma` (LR decay) on every run → wrong LR schedule | token `__GAMMA__`, only `model.gamma` substituted; guard asserts `scheduler.gamma`=0.1 |
| **fabricated config schema** (`name:`/`manifest_dir:`/`${oc.env}`) | crashes in `read_config`/`get_model`; looks plausible | migrated to real schema; `_paths.sh` guard rejects un-migrated templates |
| CBM(inception,27M) vs MCBM(resnet18,12M) | 2.3× capacity gap confounds CBM-vs-MCBM | same-backbone spine (C.2) |
| sbatch: conda not initialized; `set -u` vs conda `_CE_M` | job dies before training | robust activation + `set +u` around it |
| notebook written to `curated/curated/…` | wrong path | moved; species-constancy folded into existing `notebooks/01` |

---

## E. Known limitations / open items

- ❌ FunnyBirds test = **10 images/species** — recall-gap underpowered (→ appendix).
- ❌ `cub70-*` configs still fabricated schema (guard blocks them until migrated).
- ⚠️ CUB `n_groups_concepts: 28` unverified (must equal #attribute groups or the
  loader raises).
- ⚠️ Env actually used is `cubvision-gpu`, not the shipped `environment-mcbm.yml` —
  snapshot `conda list` of the real env for provenance.
- ⚠️ Generated results written inside `external/` (working-tree pollution).
- ❌ Backbone capacity-robustness (resnet18 vs resnet50) not yet run.
- ❌ Final val→test selection+eval script (picks best-val epoch, evals real test) —
  to be built with the eval-table dump step.

---

## F. Reproducibility anchors

- Upstream SHAs: `external/COMMITS.txt`.
- Seeds: `-s` per run (≥3 for error bars on any "X vs gamma").
- Deterministic config generation from templates via `train/_paths.sh: gen_config`.
