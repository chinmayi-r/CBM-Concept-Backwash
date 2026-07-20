# Handoff — honest state, for another model or person

Written 2026-07-20. This is a **fresh** handoff (the older `HANDOFF.md` is stale — it
predates training). Full raw conversation log is committed at
`curated/SESSION_TRANSCRIPT.jsonl` (26 MB, 4838 turns) if you want the literal history.

Branch: `claude/cbm-mcbm-validation-curated-efkd4y`. Cluster data root:
`$CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data`.
Run `bash curated/status.sh` for a live inventory of jobs/data/models.

---

## 1. What the project is (plain)
FunnyBirds is a synthetic dataset: 50 bird species, each built from parts (beak/eye/wing/
foot/tail) with a handful of variants each; 26 part-"concepts". By construction **concept =
f(species)** (every bird of a species has the same parts). Two models are trained on it: a
standard **CBM** (concept-bottleneck) and an **MCBM** (adds an information-bottleneck
"minimality" penalty, strength = γ).

**The question:** is each concept detector *grounded* (it reads its own part's pixels) or
*backwashed* (it reads species identity instead)? E.g. does the "tail" detector decide
"tail_type_3" by looking at the tail, or by recognising the whole bird is species-7?

**The test:** re-render an image with **one part swapped** to another species' variant. A
grounded detector follows the new part; a backwashed one keeps reporting the original
species' part. Also **deletion** (remove a part; does its concept still fire?).

---

## 2. Findings so far — with honest caveats (do NOT overstate these)
Most γ have **1 seed**; only MCBM γ=0 and γ=0.1 have 3 seeds. Treat everything as
**provisional**. Numbers live in `RESULTS.md`.

1. **On untouched images, both models look grounded.** An absent part's concept sits
   strongly negative. Backwash is **intervention-only** — it appears only under swap/deletion.
2. **Under swap, the tail concept is species-anchored.** It tracks the swapped-in tail only
   ~20–37% of the time (below the 50% coin flip); foot/wing track ~85–95%.
3. **It is graded, NOT tail-only.** On CBM, beak ~46–54% and eye ~57–63% also fail a lot.
   Tail is worst, but the phenomenon is a gradient across parts. (Earlier drafts overclaimed
   "tail-only" — corrected.)
4. **Deletion:** tail retained_frac ≈ 0.16 (CBM) — *modest* but the highest of any part.
5. **Minimality (γ) does NOT fix it.** Tail grounding is flat-to-slightly-worse as γ rises.
   Important: the γ penalty **saturates by γ=0.1** (its loss term drops 430→~2 and stays),
   so the "sweep" is effectively **off (γ=0) vs on (γ≥0.1)**, not a smooth dial — any
   "trend vs γ" past 0.1 is nearly flat by construction.
6. **Downstream impact is tiny.** Even when a concept is fooled, the species *prediction*
   barely moves (P(donor species) rises only ~0.05 on CBM / ~0.12 on MCBM at large margin).
   **So backwash breaks the model's stated REASONS, not its accuracy.** This is the honest,
   load-bearing framing. Do **not** claim it breaks predictions.
7. **Species probe:** the full 26-concept vector predicts species ~0.99; any single part's
   concepts predict it weakly (tail 0.19, chance 0.02). The species code is in the *assembled*
   vector, not one part.

### Confounds NOT yet resolved (important)
- **Training-occlusion.** The tail is the smallest / most-often-occluded part. Low tail
  grounding could be *under-training* (rarely seen clearly) rather than species-anchoring.
  The **relabeled (RL)** model was trained specifically to break concept↔species while
  holding visibility fixed — **its swap has not run yet** (see §4). That swap is the causal test.
- **Non-independent points.** Swap scatter plots reuse each image's activation across its
  swaps, so density is inflated — read *positions*, not density.

---

## 3. Current data/model state (from `status.sh`, 2026-07-20)
- **Trained models:** CBM (seeds 1-3), MCBM g0 & g0.1 (1-3), g0.3/g1/g3/g5 (seed 1),
  vanilla (1-3), **RL g0–g5 (seed 1, epoch 100 — done)**. Junk dir `funnybirds-mcbm-g30`
  (no saved epochs) can be deleted.
- **Swaps done:** CBM s1; MCBM all 6 γ at s1 (g0/g0.1 also s2,s3). **RL swaps: none yet.**
- **Species probe + deletion grounding:** all standard configs (13 files each).
- **RL relabeled data:** built.
- **Nothing running** (only a Jupyter session).

**The ONE remaining compute step:** the RL swap (§4).

---

## 4. The one command left to run
```bash
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
CONFIG_PREFIX=funnybirds-mcbm-rl GAMMAS="0 0.1 0.3 1 3 5" SEEDS="1" sbatch train/renderer_swap.slurm
```
Fills `swap/funnybirds-mcbm-rl-*.csv`, which notebook `03rl` reads. If the relabeled tail
grounds → the species-level *labels* caused the backwash. If it still fails → it's deeper
(bottleneck / architecture). This is the causal payoff.

---

## 5. File map
**Curated notebooks (run on the new data; condensed, objection-first):**
- `notebooks/02_funnybirds_cbm.ipynb` — CBM analysis.
- `notebooks/03_funnybirds_mcbm.ipynb` — MCBM γ sweep.
- `notebooks/03rl_funnybirds_mcbm_relabeled.ipynb` — RL causal test (waiting on RL swap).
- `notebooks/04_cub_analysis.ipynb` — raw CUB label analysis (the label-standardization critique).

**Original reference notebooks (`funnybird_notebooks/`)** — the FULL versions, every plot &
table, nothing condensed. **They do NOT run on curated** and it's not just the model-loading:
they load an old checkpoint format + precomputed feature dumps + an old `datasets.
funnybirds_dataset` module, and several *body* cells break too. Making them runnable on the
new models is the user's requested next task (§6).

**Docs:** `RESULTS.md` (number log), `DECISIONS.md`, `STORY.md`, `REFERENCE_CELL_MAP.md`
(MCBM ref→curated cell mapping), `CBM_BASELINE.md` (CBM ref→curated), `CATCHUP.md`
(task list + a per-plot objection-first critique + full catch-up plan), `status.sh`
(one-shot state check). Review page: `figs/cbm_review.html` (every CBM figure shown +
critiqued + a plain-terms glossary).

---

## 6. What the user actually wants next (and how to work with them)
- **Make the FULL original notebooks runnable on the new models** — as **SEPARATE files**
  (do **not** overwrite the originals or curated 02/03; put copies e.g. under
  `notebooks/original_runnable/`). **Check every cell**, not just setup — body cells break too.
  This needs iterative run-and-fix with the user, because the model can't execute on the
  cluster (no GPU/renderer here) — so land the setup shim + obvious body fixes, then have the
  user run and paste errors, fix, repeat.
- **The known-hard-to-port piece:** the "GT ceiling" plots (set a concept to 1/0 and read the
  label head) — minimal_cbm's label head reads the *bottleneck*, not the 26 concepts, so that
  intervention can't be computed the old way. Degrade those to NaN and note it.
- **Do not overstate.** The user has repeatedly (correctly) caught overstated captions.
  Hedge; single-seed = provisional; say "supports/consistent with," not "proves."
- **Move efficiently.** The user is frustrated by cycles of tweaking outputs. Prefer landing
  runnable artifacts over polishing.

---

## 7. Next steps, ordered
1. **Run the RL swap** (§4) → re-export `03rl` → the causal verdict.
2. **Adapt the original notebooks** to run on curated models, as separate files, cell-by-cell,
   verifying each (iterative with the user).
3. (Optional) seed parity — swaps for g0.3/g1/g3/g5 seeds 2-3 already partly submitted; check.
4. **Honest write-up** using §2 with all its caveats.

If you're a fresh model picking this up: run `bash curated/status.sh`, read `RESULTS.md` and
this file, then start at step 1 or 2. Don't re-derive; the data is mostly done.
