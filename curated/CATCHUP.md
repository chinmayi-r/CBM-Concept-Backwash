# Catch-up: tasks, what to run, every-plot analysis, full plan

## A. Every task you gave me (checklist)

| # | task | status |
|---|------|--------|
| 1 | fwd/bwd split heatmap in nb03 | ✅ done (§6a) |
| 2 | replace the confetti with a readable plot | ✅ done (grouped box) |
| 3 | exporter to dump every figure as PNG for review | ✅ `analysis/export_figs.py` |
| 4 | fix `*.png` gitignore so figs push | ✅ done |
| 5 | load ALL seeds+γ (not just -s1) | ✅ `load_swaps` fixed |
| 6 | say which γ/seeds finished in the timed-out job (no rerun) | ✅ §B below |
| 7 | commands to finish incomplete + push back | ✅ §B below |
| 8 | analyse EVERY plot one-by-one, critic-first | ✅ §C below (nb03 + nb02) |
| 9 | full CBM (nb02) too, not just MCBM | ✅ §C.2 |
| 10 | comprehensive catch-up plan: all seeds/γ, CBM+MCBM+RL, extra plots for the holes, no info lost | ✅ §D below |
| 11 | species probe for all γ | ⏳ command ready (§B); bug already fixed |
| 12 | build the RL (relabel→retrain→swap) notebook pipeline | ⏳ planned §D.3 |

---

## B. What finished, and the ONLY commands to run (nothing done gets rerun)

The overnight job **3308851 timed out at the 12 h wall**, mid-g1. From the CSVs on disk:

| γ | seeds with a loadable combined CSV |
|---|---|
| 0 | s1, s2, s3 ✅ |
| 0.1 | s1, s2, s3 ✅ |
| 0.3 | s1 ✅ |
| 1 | **partial** — beak/eye/foot/tail done, **wing missing**, no combined |
| 3 | **none** |
| 5 | s1 ✅ |
| CBM | s1 ✅ |

The driver is **resumable** (`if part_csv.exists(): reuse & continue`). So finishing g1 only computes the missing **wing** (~20 min), not the whole thing.

```bash
export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
cd ~/CBM-Concept-Backwash/curated

# 1) finish the gamma axis at 1 seed (g1 resumes -> only 'wing'; g3 runs full ~2h)
CONFIG_PREFIX=funnybirds-mcbm GAMMAS="1" SEEDS="1" sbatch train/renderer_swap.slurm
CONFIG_PREFIX=funnybirds-mcbm GAMMAS="3" SEEDS="1" sbatch train/renderer_swap.slurm

# 2) species probe for EVERY gamma (renderer-free, cheap; the skip-bug is fixed)
bash analysis/grounding_sweep.sh

# 3) once (1) finishes, refresh + push the figures
git pull origin claude/cbm-mcbm-validation-curated-efkd4y
python analysis/export_figs.py notebooks/03_funnybirds_mcbm.ipynb figs/03
python analysis/export_figs.py notebooks/02_funnybirds_cbm.ipynb figs/02
git add figs && git commit -m "figures: full gamma axis + all seeds" && \
  git push origin claude/cbm-mcbm-validation-curated-efkd4y
```

After this: **all 6 γ present at ≥1 seed**, g0/g0.1 at 3 seeds, species probe for all γ.

---

## C. Every plot, one by one — reading + the objection first

Convention: **O:** = the objection a critic would raise; **R:** = what the plot actually supports.

### C.1 — nb03 (MCBM)

1. **Overall retention vs γ.** Flat ~0.08–0.11, ≈ CBM. **O:** overall averages the 4 grounded parts, so it's near-0 no matter what — uninformative alone. **R:** baseline only; the per-part view (2) is the real one.
2. **Per-part retention vs γ.** tail ~0.28–0.45 with a **wide seed band (0.2–0.5)**, others ≈0. **O:** the band is wide and the curve is non-monotone — the γ=0.3 bump is inside the noise; do **not** read a trend. **R:** tail stays high & flat; minimality doesn't bring it down.
3. **Species probe vs γ.** One point (γ=3): c_preds ≈0.99, tail-concepts ≈0.27. **O:** one point can't show a γ trend. **R:** at γ=3 the concept layer is a species code; the across-γ claim is *pending* the probe re-run.
4. **Did γ bite?** rep_loss 430→2 **by γ=0.1**, mean|z| 18→3; task/concept flat. **O (important):** the knob **saturates at γ=0.1** — γ=0.1…5 are nearly identical compression. So this is really **off (γ0) vs on (γ≥0.1)**, not a graded 6-point sweep. Every "vs γ" trend past 0.1 is a near-flat line, by construction.
5. **IB compresses z.** z-std 20→2.5; right panel: tail retention vs z-std shows no relationship. **O:** only γ=0 has high z-std; all γ≥0.1 pile at z-std≈2.5 — the "scatter" is one point vs a blob, so no trend can be fit. **R:** compression jumps once and grounding does not follow — still valid, but it's a 1-point-leverage statement.
6. **Combined heatmap (part×γ).** tail 0.36/0.24/0.20 (orange→red), foot/wing green. Clean; tail worst, reddens with γ.
7. **fwd/bwd split.** tail red in BOTH (fwd 0.34→0.16, bwd 0.38→0.23). **R:** symmetric backwash — not the reference's 0.50 averaging artifact.
8. **Does minimality fix the tail swap?** 0.36→0.20, below CBM ref and 0.5. **O:** 3 points, and the drop is all at γ0→0.1 (the saturation again). **R:** directional "not fixed, if anything worse."
9. **§E dots+box.** tail box below 0 at every γ; foot median well above. Good.
10. **Occlusion control (tail vs foot).** visible-only lifts tail only ~0.36→0.42 at γ0 (less at higher γ); tail stays ~0.2–0.4 ≪ foot 0.87–0.96. **R:** occlusion does not rescue tail.
11. **z-ordering per part vs γ.** tail lowest at every γ. Restates 6.
12. **CBM vs MCBM heatmap.** tail column 0.37 (CBM) / 0.36 / 0.24 / 0.20 — reddens with γ; foot green throughout. **R (headline):** neither sigmoid (CBM) nor minimality (any γ) grounds the tail; γ makes it slightly worse.
13. **Mean margin per part vs γ.** tail ≈/below 0, others positive. Restates.
14. **Violation heatmap.** tail darkest. Restates (1−ordering).
15. **Margin box per part×γ.** tail boxes below 0, others above; γ doesn't lift tail. Good.
16. **Grounding before swap (γ0).** donor-absent z sits at −10…−40 while source-present z is +5…+45; all parts below the diagonal, tail lowest. **O:** the vertical/horizontal streaks are the **reused-activation artifact** (each original's z reused across swaps) — read positions, not density. **R:** strong ON-distribution grounding → the backwash is **intervention-only**, which is the honest framing (the model looks perfect until you swap).
17. **Downstream: margin → P(donor species).** binned mean rises from ~0 to only **~0.12**; a thin cloud jumps to P=1 past margin≈5. **O (important):** the effect is **small** — even at positive margin, mean P(donor) ≈0.12, i.e. the concept-layer failure usually does **not** flip the class output. **R:** backwash is real in the concept layer but only weakly propagates downstream — don't over-claim task-level impact.
18. **Per-source-species tail violation.** sorted 1.0→0.3; many species 100% violation. Non-uniform → species-specific.
19. **Top-20 concept slots.** tail_0…tail_5 dominate the top. Good.
20. **Tail visibility vs margin / by outcome.** violations span all pixel counts; the "correct" (green) histogram is a tiny sliver even at high visibility. **R:** even the most-visible tail swaps mostly violate → not occlusion.
21. **z distribution MCBM γ0 vs CBM.** CBM tight bimodal [−10,10]; MCBM γ0 broad [−40,40]. **R:** γ0 ≠ CBM scale → keep separate notebooks.

### C.2 — nb02 (CBM)

1. **Training curve.** task ~0.75, concept ~0.99, plateau → epoch 100 safe.
2. **Deletion retention.** tail visible-only **0.16 ±0.04**, others 0.01–0.04; all-removals inflated (tail 0.35). tail clearly highest, but 0.16 is modest — state it as modest.
3. **Species probe.** ~0.99; tail carries most species info of any part.
4. **Line-them-up (retention vs species-code / n_variants).** tail top-right on both. **O:** 5 points with tail as an extreme outlier — the "correlation" is one-point leverage, not a trend. **R:** consistent with the mechanism, not proof.
5. **Per-species leakage.** non-uniform (median 0.10, max 0.66). Species-specific.
6. **On-distribution grounding.** source-present +15…+20, donor-absent −18…−36 (tail most negative, −36). **R:** strong on-dist grounding — tail is *most confidently "off" when absent* yet *most backwashed under swap* = context-dependent species reading. Good control; backwash is intervention-only.
7. **fwd/bwd z-ordering.** tail 0.35/0.37, wing 0.87/0.82, beak 0.43/0.54, foot 0.87/0.85, eye 0.57/0.63. tail lowest, symmetric.
8. **Per-swap margin dots.** tail cloud below 0. Mirrors.
9. **Inspection grid.** only the target part changes across orig/swap/delete; camera/bg fixed → failures aren't rendering artifacts.
10. **Margin box per part.** tail **63% viol** (box below 0), wing 16%, beak **51%**, foot 11%, eye **40%**. **O (important):** beak (51%) and eye (40%) are **also substantially violated** — the story is NOT "only tail." tail worst, but beak/eye are partially backwashed. Don't over-claim tail-exclusivity.
11. **Grounding vs visibility, all parts.** tail stays ~0.3–0.6 across pixel bins, peaks ~0.6 at 100–200 px, never reaches foot ~0.9; **beak/eye climb** with visibility. **O:** for beak/eye visibility DOES matter (partly occlusion) — so the "not occlusion" claim is clean for **tail** but weaker for beak/eye. **R:** tail's failure isn't occlusion; beak/eye are mixed.
12. **Occlusion filter (before/after).** filtering low-viz raises tail modestly, still ≪ grounded parts.
13. **Tail confusion matrix.** diagonal dim; **column 2 bright** — variants 0,1,2,7 all fire tail_2 → a "default tail concept." **O:** argmax attribution ≠ the ordering metric, single seed, and the diagonal isn't fully dark. **R:** a *hypothesis* (default-concept collapse), not proof; needs the all-part matrix (driver now logs it) to show grounded parts are clean by contrast.
14. **Grounding-before-swap scatter.** donor hugs its floor. Mirrors on-dist grounding.
15. **Top-20 concepts.** tail variants dominate.
16. **Per-source-species violations.** non-uniform.
17. **Visibility per part.** tail smallest/most-occluded — motivates the occlusion control.
18. **Downstream margin → P(species).** small effect (mirrors nb03 #17): concept failure weakly propagates.

### C.3 — cross-cutting objections to carry into the writeup
- **γ is effectively binary (off vs on).** The knob saturates at γ=0.1, so every "vs γ" curve past 0.1 is flat *by construction of the loss*, not because minimality plateaus in effect. Say this explicitly or a referee will.
- **Downstream impact is weak.** Backwash lives in the concept layer; it mostly does not flip the class. The claim is "un-grounded concepts," not "wrong predictions."
- **Not tail-exclusive.** beak (51%) and eye (40%) also violate a lot on CBM. tail is worst + most-occluded, but the phenomenon is graded across parts.
- **On-distribution the models look grounded.** Both CBM and MCBM turn the absent concept strongly OFF. The finding is *intervention-only*, which is the honest and stronger framing.
- **Non-independent dots.** swap scatters reuse each original's activation → density is inflated; every scatter caption should say "read positions, not density." (Already partly done.)

---

## D. Full catch-up plan — all seeds, all γ, CBM+MCBM+RL, holes filled, no info lost

### D.1 Data to generate (SLURM; per-(γ,seed) jobs so nothing hits the 12 h wall; all resumable)
- **MCBM swap, 6 γ × 3 seeds.** Have: g0(3), g0.1(3), g0.3(1), g5(1), g1(partial), g3(0).
  Run: `for g in 0.3 1 3 5; do for s in 1 2 3; do CONFIG_PREFIX=funnybirds-mcbm GAMMAS="$g" SEEDS="$s" sbatch train/renderer_swap.slurm; done; done`
  (already-done (γ,seed) resume instantly; only missing parts compute.)
- **CBM swap seeds 2,3** (for parity): `for s in 2 3; do CONFIG_PREFIX=funnybirds-cbm SEEDS="$s" sbatch train/renderer_swap.slurm; done`
- **Species probe, all γ, all seeds:** `bash analysis/grounding_sweep.sh` (fixed).
- **Deletion grounding, all seeds:** already all 6 γ; add seeds if training has them.

### D.2 Plot holes to close (from REFERENCE_CELL_MAP, without removing anything)
- **All-part confusion matrix** (currently tail-only). Driver already logs `z_cf_{part}_{i}` for every part → re-run swap, then add a per-part confusion grid so a grounded part's clean diagonal contrasts the tail collapse. (nb02 §8 + nb03.)
- **fwd/bwd margin scatter** (ref 68): mean fwd-margin vs bwd-margin per (γ,part); a species encoder sits on the anti-diagonal. One small cell.
- **Per-variant margin histograms** (ref 54): optional granular view once seeds give enough per-panel n.
- **Per-γ z before/after overlay** (ref 84): donor z in original vs counterfactual, per γ.
- **beak/eye honesty panel:** promote beak(51%)/eye(40%) into the main story as "graded, not tail-only."
- **Downstream-impact caveat panel:** make the small P(donor) effect explicit so we don't over-claim task-level harm.

### D.3 RL (relabeled) — the one real build, not just a run
Pipeline (new): `analysis/relabel_funnybirds.py` (concept=0 when the part is <5% of the species-median pixel area) → a `funnybirds-mcbm-rl-g*` train config → train → swap (same driver, `--config-prefix funnybirds-mcbm-rl`) → populate `03rl`. This is the **causal disentangler** for the training-occlusion / label-correlation confound: if tail grounding recovers under relabeling, labels were the lever; if not, it's deeper. This is the only genuinely new code in the catch-up; everything else is runs + plot cells.

### D.4 Order of operations
1. Run §B now → 6-γ axis complete, plots refresh (biggest visible payoff).
2. Kick the full 3-seed grid (D.1) — background, resumable.
3. Add the D.2 plot cells (they render from existing CSVs).
4. Build D.3 (RL) — the one new pipeline.
5. Final one-by-one pass on the *complete* plots with the C.3 objections baked into every caption.

---

## E. STANDING TASK — final full review (do AFTER catch-up is complete)

Explicit instruction (do not skip, do not sample): once all γ (incl. g3), all seeds, CBM,
MCBM, and RL data are in and the plots are final, go through **every single figure that
notebook 02 and notebook 03 print — every one, regardless of whether it was shown before**
and for each one:
1. **Print** it (open the PNG and actually look).
2. **Criticise** — state the strongest objection a referee/critic would raise first.
3. **Consider** — weigh whether the objection kills it, weakens it, or is answered.
4. **Final opinion** — a one-line verdict: keep as-is / fix / cut / needs more data.
5. **Compare to the paper story** — does this figure support, complicate, or contradict the
   narrative we're building? Flag any figure that overclaims relative to what it shows.

Carry the C.3 cross-cutting objections into this pass (γ is binary off/on past 0.1;
downstream impact is weak; not tail-only — beak/eye also violate; on-distribution both
models look grounded so the finding is intervention-only; swap scatters have non-independent
points). Output a single consolidated verdict table (figure → objection → verdict → story fit).
The paper story to test each figure against lives in `STORY.md` / `RESULTS.md`.
