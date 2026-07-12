# RUNBOOK — what's done on adroit, what to run to finish

Per-claim: **(a) one command to CHECK if it's already done**, **(b) the command to RUN it.**
Claims map to STORY.md §8 (C1–C6). Results land in RESULTS.md.
Everything runs from `curated/` with `CURATED_DATA` exported.

```bash
# ---- session setup (adroit login node) ----
cd .../CBM-Concept-Backwash/curated
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
module load anaconda3/2025.6 ; set +u ; conda activate cubvision-gpu ; set -u
MCBM=external/minimal_cbm            # trained checkpoints live here
```

---

## 0. One-glance status board (run these five, read the answers)

```bash
squeue -u cr7998                                              # anything still running?
ls $MCBM/results/funnybirds-*/*/models/epoch_100.pt 2>/dev/null   # which models finished (final ckpt)
ls $CURATED_DATA/grounding/*.parquet 2>/dev/null             # C1/C2 deletion-backwash scored
ls $CURATED_DATA/species_probe/*.json 2>/dev/null            # C4 species-probe scored
ls $CURATED_DATA/backwash_vs_gamma.* 2>/dev/null             # C2 final figure
ls $CURATED_DATA/data_analysis/SUMMARY.txt 2>/dev/null       # C3 dataset characterization
```

A model counts as **trained** only when `models/epoch_100.pt` exists (matches `N_EPOCHS`
in `train/configs/funnybirds-*.yaml`). A partial run just needs a resubmit — it resumes.

---

## The single fire-and-forget (does C1+C2+C4 for FunnyBirds, resumable)

If you just want it all: submit this once. It trains baselines + the γ-sweep, then
runs deletion-grounding **and** the species-probe on every checkpoint and builds the
backwash-vs-γ figure. Hits the wall → resubmit the identical line, it resumes.

```bash
GAMMAS="0 0.1 0.3 1 3 5" SEEDS="1" sbatch train/sbatch_all.slurm
squeue -u cr7998                     # watch it; tail the fb_spine_*.out for progress
```

The per-claim sections below are for checking/running pieces individually.

---

## C1 — CBM deletion backwash on FunnyBirds  (tail retains a removed part)  · STORY §6

The headline: for each present part, delete it (pre-rendered intervention image) and
read the retained prob of the species-typical concept. tail≈0.36, others≈0. **DONE**
(🟡 provisional in RESULTS.md).

```bash
# CHECK
ls $CURATED_DATA/grounding/funnybirds-cbm-s1.parquet
# RUN (single model)
python3 analysis/grounding_deletion.py --config funnybirds-cbm --seed 1 \
    --funnybirds-root $CURATED_DATA/FunnyBirds \
    --pkls $CURATED_DATA/funnybirds_processed \
    --out $CURATED_DATA/grounding/funnybirds-cbm-s1.parquet
```

---

## C2 — MCBM γ-sweep: does minimality fix grounding?  (backwash vs γ)  · STORY §6 / §1b

The foil. Same backbone/data as CBM; only the head + γ change (effective IB force =
γ×0.2). Prediction: backwash persists / does not vanish across γ. Needs the γ models
trained, then grounding on each, then the collect step → `backwash_vs_gamma.{csv,png,pdf}`.

```bash
# CHECK — which gammas are trained + whether the figure exists
ls $MCBM/results/funnybirds-mcbm-g*/1/models/epoch_100.pt 2>/dev/null
ls $CURATED_DATA/backwash_vs_gamma.png
# RUN — train the sweep + score + collect (idempotent, skips finished gammas)
GAMMAS="0 0.1 0.3 1 3 5" SEEDS="1" bash train/run_all_funnybirds.sh
# or just re-score + re-collect if models already trained:
bash analysis/grounding_sweep.sh
```

Gamma values are the paper's own (CUB 0.05–0.3, synthetic 1–5); γ=0 is MCBM-with-no-IB,
**not** vanilla CBM (different head — see DECISIONS §"γ=0≠CBM").

---

## C3 — Dataset characterization: concept=f(class), part features  · STORY §3b / §6

Renderer-free, CPU, seconds. Class balance, prevalence, class×concept matrix (fraction
of cells exactly 0/1 → clean species→concept lookup), species-constancy (why the recall
gap is a CUB axis not a FunnyBirds one), and the neutral per-part candidate features
(n_variants / visibility) to line up against measured backwash.

```bash
# CHECK
cat $CURATED_DATA/data_analysis/SUMMARY.txt
# RUN
python3 analysis/data_analysis.py \
    --funnybirds-root $CURATED_DATA/FunnyBirds \
    --pkls $CURATED_DATA/funnybirds_processed \
    --out $CURATED_DATA/data_analysis
```

Notebook equivalent (for the write-up figures): `notebooks/01_data_analysis.ipynb`.

---

## C4 — Species-identity probe on the bottleneck (the mechanism)  · STORY §6b

Ports `fb_cbm_counterfactual.ipynb §6`, renderer-free. Linear probe species←z and
species←c_preds (chance 1/50), plus per-part concept-block probes. High species←c_preds
= the reported concepts alone pin the species → a part concept is answerable by
class-lookup (the standing opportunity for backwash). Read together with C1's per-part
deletion result = the mechanism.

```bash
# CHECK
cat $CURATED_DATA/species_probe/funnybirds-cbm-s1.json
# RUN (single model; the sweep runs it on all checkpoints automatically)
python3 analysis/species_probe.py --config funnybirds-cbm --seed 1 \
    --funnybirds-root $CURATED_DATA/FunnyBirds \
    --pkls $CURATED_DATA/funnybirds_processed \
    --out $CURATED_DATA/species_probe/funnybirds-cbm-s1.json
```

---

## C5 — CUB70: occlusion grounding on REAL birds  (WIP)  · STORY §7 / roadmap §9

CUB70 = first 70 CUB classes that ship per-part segmentation masks → part-level
grounding on real images (full CUB has no masks). **Status: scaffolding, not yet run.**
`train/cbm_cub70.sh` / `mcbm_cub70.sh` filter the pickles to classes 0–69 and support
`original`|`relabeled` labels (for C6). The mask-based visibility + occlusion grounding
scripts (`data/cub70/build_cub70_visibility.py`, a CUB70 occlusion probe) are the
remaining build. Take the current cub70 configs with a grain of salt — verify the
schema against a real minimal_cbm CUB run before trusting a training launch.

```bash
# CHECK
ls $CURATED_DATA/runs/cub70_cbm_original_seed1 2>/dev/null
ls $CURATED_DATA/cub70_visibility.parquet 2>/dev/null
# RUN (once verified)
bash train/cbm_cub70.sh 1 original
bash train/mcbm_cub70.sh 1 original
```

Notebook: `notebooks/02_cub_data_analysis.ipynb` (DATASET selector, §5 = CUB70 masks).

---

## C6 — Visibility-aware relabeled CBM  (Δ backwash after cleaning labels)  · STORY §8

Train CBM on labels corrected for part visibility (a bird gets a part-concept only
when the part is actually visible), compare grounding vs the original class-level
labels. Directly tests the CBM label-standardization critique (§3b). Depends on C5's
CUB70 masks + `relabel_cub_with_cub70.py`. **Status: pending C5.**

```bash
# CHECK
ls $CURATED_DATA/CUB_processed/class_attr_data_10_relabeled 2>/dev/null
# RUN (after C5 masks exist)
python3 data/cub70/relabel_cub_with_cub70.py          # build relabeled pickles
bash train/cbm_cub70.sh 1 relabeled                   # then re-score grounding, compare
```

---

## Finishing order (shortest path)

1. **C2** — submit `sbatch_all.slurm` with `GAMMAS="0 0.1 0.3 1 3 5"`. This alone
   closes C1(done)+C2+C4 for FunnyBirds and produces `backwash_vs_gamma.png`.
2. **C3** — run `data_analysis.py` (seconds); lock the concept=f(class) / part-feature numbers.
3. Send me the outputs (paste the SUMMARY.txt / the .json / push the figure) → I firm up
   RESULTS.md 🟡→🟢 and update STORY.md where a number moves the narrative.
4. **C5 → C6** — CUB70 masks + occlusion grounding, then relabeled CBM. Verify the
   cub70 config schema first; ping me and I'll finish the mask/occlusion scripts.
