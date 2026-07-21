# PLAYBOOK — the only file you follow

When you don't know what to do: **come here, find the current STEP, run its commands,
paste the 📋 REPORT back to Claude.** Steps are in order. Don't skip.

---

## Who does what (so you never wonder "do I push?")

| | You (on adroit) | Claude (here) |
|---|---|---|
| **code / notebooks / scripts** | `git pull` only — **never edit or push code** | edits + **pushes** to the branch |
| **running jobs & notebooks** | run on the cluster | — |
| **results** (parquets, csv, json, figures) | live in `$CURATED_DATA` (scratch) — **not git**; you **paste** them back | reads what you paste, updates STORY/RESULTS/DECISIONS, pushes |

**The loop:** `git pull` → run the STEP → 📋 paste output → Claude updates docs + pushes → `git pull` → next STEP.
You only ever type `git pull --rebase`. Claude does all pushing of code.

## Start of EVERY session (copy-paste block)
```bash
cd /scratch/network/cr7998/cv_emergence_project/curated
module load anaconda3/2025.6 && conda activate cubvision-gpu
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
git pull --rebase
```

## What the files are (reference)
- `notebooks/0X_*.ipynb` — what you SHOW people. Run them, they render figures+text.
- `analysis/*.py`, `train/*.sh` — cluster plumbing that MAKES the data. You don't read these.
- `RESULTS.md` — the measured-numbers log. `STORY.md` — the paper narrative. `DECISIONS.md` — why.
- `RUNBOOK.md` — per-claim command reference (this PLAYBOOK is the ordered version).

---

# PHASES (checkbox — where the whole project is)

- [x] **A. FunnyBirds · CBM** — deletion backwash (tail 0.36) + species probe. DONE.
- [~] **B. FunnyBirds · MCBM γ-sweep** — backwash vs γ. Trained; **g3 diverged, retraining.**
- [ ] **C. Render notebooks 02 + 03** — the FunnyBirds story for the meeting.
- [ ] **D. Decide: is γ range enough?** (notebook 03 §4 verdict)
- [ ] **E. CUB70** — test-time visibility grounding on real birds (C5). CUB70 masks cover test images only, so they cannot support relabeled retraining.
- [ ] **F. FunnyBirds CBM-RL** — train an ordinary CBM with image-level visibility labels for the controlled relabeling test (C6).

---

# CURRENT STEPS (do these now, in order)

### STEP 1 — wait for the g3 retrain (job already submitted)
```bash
squeue -u cr7998          # when the "fb_spine" row is GONE, it's done
```
The job retrains g3, then automatically re-scores grounding + species-probe and
rebuilds the CSV/figure. While waiting you can watch: `tail -f fb_spine_*.out`
(look at the LAST `accuracy_test/task` — should climb well above 2.0).

**📋 REPORT 1** — when the job is gone, paste:
```bash
cat $CURATED_DATA/backwash_vs_gamma.csv
```
→ Claude checks whether g3 is now a real number or still NaN (diverged again).

---

### STEP 2 — render the two FunnyBirds notebooks (this is the meeting material)
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_funnybirds_cbm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_funnybirds_mcbm.ipynb
jupyter nbconvert --to html notebooks/02_funnybirds_cbm.ipynb
jupyter nbconvert --to html notebooks/03_funnybirds_mcbm.ipynb
```
Open the two `.html` files to view. (If a cell errors, paste the error — don't fix it yourself.)

**📋 REPORT 2** — paste the two printed **VERDICT** lines:
- notebook 02 §0 → "best task epoch = … / plateau vs overfit"  (settles epoch 100 vs 150)
- notebook 03 §4 → "γ moved the representation …" vs "γ barely moved …"

---

### STEP 3 — Claude acts on REPORT 2 (you just pull)
Based on §4's verdict Claude either (a) locks the FunnyBirds result into STORY/RESULTS,
or (b) wires up a wider γ sweep (10/20/50) if γ was underpowered — then pushes.
```bash
git pull --rebase
```
Then Claude tells you the next STEP.

---

## If something looks broken
- **NaN / diverged again (g3):** paste REPORT 1; Claude adds gradient clipping (a
  documented training-stability guard) and gives you a retrain command.
- **A notebook cell errors:** paste the error; Claude fixes + pushes; you `git pull` and re-run.
- **Warnings** (pymp, torchvision, sklearn deprecations): **ignore** — cosmetic, they
  never reach the notebook. Only a `Traceback`/`Error` or `NaN` matters.
