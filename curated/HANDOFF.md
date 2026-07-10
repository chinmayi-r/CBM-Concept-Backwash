# HANDOFF — where things are, and the one command to run

## TL;DR — fire this and walk away
```bash
cd /scratch/network/cr7998/cv_emergence_project
git pull origin claude/cbm-mcbm-validation-curated-efkd4y
cd curated
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
GAMMAS="0 1 3 10 30" SEEDS="1" sbatch train/sbatch_all.slurm
```
That single job does the whole FunnyBirds spine end-to-end and writes the money
figure to `$CURATED_DATA/backwash_vs_gamma.{png,pdf,csv}`. It is **idempotent**:
if it hits the 16 h wall, just run the same `sbatch` line again and it resumes
(skips anything already trained/scored). Watch it: `tail -f fb_spine_*.out`.

## What the job does (train/run_all_funnybirds.sh)
1. **Baselines** vanilla + CBM (skipped — already trained).
2. **MCBM gamma sweep** at γ ∈ {0,1,3,10,30}, 100 epochs each (~2 h/run on the A100
   MIG). This is the long part (~10 h serial). The foil curve.
3. **Deletion grounding** on every CBM/MCBM checkpoint (analysis/grounding_deletion.py):
   remove each part, measure whether the model still predicts it → **backwash rate**.
4. **Collect + plot** `backwash_vs_gamma` (analysis/collect_backwash.py): MCBM curve
   + CBM/vanilla reference lines.

## How to read the result when you're back
`$CURATED_DATA/backwash_vs_gamma.png` — x = γ (log), y = backwash (retained P of a
REMOVED part; 1 = model "sees" absent parts = species-lookup, 0 = grounded).
- **Curve stays high across γ** ⇒ the thesis: even maximal minimality (MCBM) doesn't
  fix concept–class backwash. That's the paper.
- Curve drops as γ rises ⇒ minimality partially grounds concepts (still a result).
`backwash_vs_gamma.csv` has per-model numbers (p_intact, p_removed, backwash).

## If you want the finer grid / error bars later
`GAMMAS="0 0.3 1 3 10 30" SEEDS="1 2 3" sbatch train/sbatch_all.slurm` (much longer;
resumable). More seeds → error bars on the curve.

## Faster grounding-only (if training is already done)
Grounding is tiny and needs no GPU:
```bash
salloc --cpus-per-task=4 --mem=16G --time=00:30:00
module load anaconda3/2025.6 && conda activate cubvision-gpu
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
cd .../curated && bash analysis/grounding_sweep.sh
```

## State of the world (2026-07-10)
- Pipeline validated end-to-end on the official code (CBM trained: task ~75%,
  concept ~99.6%). All infra bugs fixed (see DECISIONS §D).
- Trained: `funnybirds-vanilla` (epoch 200), `funnybirds-cbm` (epoch 150). MCBM
  sweep + grounding = what the job above produces.
- The science, design, paper-gap arguments, and the two off-manifold tests
  (deletion = cleanest, novel-combo swap = next) are recorded in `DECISIONS.md`
  (§B.1 theory, §D.1 narrative, §D.2 reconciling CBM/MCBM figures, §D.3 tests) and
  `PAPERS.md` (source-paper methods + what we borrow).

## Next experiments after the money plot (in order)
1. **Novel-combo swap** (the other off-manifold test): build the swap-graph from
   `classes.json`, render/score swaps that land OFF the 50-species manifold,
   measure concept accuracy on the swapped part. (Deletion is done; swap is the
   complement.)
2. **CUB**: train vanilla/cbm/mcbm (configs migrated, `dataset cub`), run the
   recall-gap indicator (CUB is powered where FunnyBirds isn't). Verify
   `n_groups_concepts: 28` first (DECISIONS §E).
3. **The "so what" figure**: compositional-generalization failure — concept
   accuracy on novel combinations vs on-distribution (DECISIONS §D motivation).
4. Optional: `ARCH=resnet18` capacity-robustness pass.
