# New-chat handoff — FunnyBirds CBM/MCBM/RL and CUB

Read this file first. It supersedes status claims in older handoffs. Do not infer
that every old result is invalid, and do not rewrite conclusions before inspecting
the executed HTML figures.

## 1. Research progression

The intended story is:

1. **Non-RL FunnyBirds data notebook:** understand the dataset and the fact that
   concepts are highly tied to species.
2. **Standard CBM notebook:** discover whether concept scores are grounded in their
   own part pixels.
3. **Standard MCBM notebook:** test whether the minimality/information-bottleneck
   penalty fixes the grounding problem.
4. **RL notebook:** only after the phenomenon is established, test one proposed
   cause: species-level concept labels remain positive when a part is hidden.
5. **CUB/CUB70:** test related visibility/grounding behavior on real birds.

RL is a causal follow-up. It does not replace the non-RL notebooks.

## 2. What the existing non-RL results support

These are the current honest conclusions, subject to the caveats below:

1. Untouched FunnyBirds test accuracy and concept accuracy are very high.
2. In the standard CBM swap run, tail follows the replacement much less often
   than foot/wing. Beak and eye are also imperfect: the effect is graded, not
   tail-only.
3. Standard CBM tail deletion retention is about 0.16: modest, but the highest
   part-level retention.
4. The full concept vector predicts species strongly; a single part group predicts
   species much more weakly. Species information is mainly distributed across the
   assembled concept vector.
5. MCBM minimality does not obviously repair tail grounding. Do not claim a smooth
   gamma trend: the penalty largely saturates by gamma 0.1, high gamma usually has
   one seed, and legacy cross-gamma swaps were independently rendered.
6. The downstream species-probability effect is small. The strongest demonstrated
   harm is unreliable concept explanation, not widespread task prediction failure.
7. Occlusion alone did not rescue the tail in the existing analyses, but any plot
   joining a live RGB render to a separately rendered part map must be rechecked
   with the fixed-render evaluation before being called conclusive.

Reference files:

- `notebooks/01_funnybirds_analysis.ipynb`
- `notebooks/02_funnybirds_cbm.ipynb` and executed HTML
- `notebooks/03_funnybirds_mcbm.ipynb` and executed HTML
- `EXPERIMENT_TRACKER.md` — accepted/provisional/quarantined work and job ledger
- `CATCHUP.md` — per-figure objection-first review
- `CBM_BASELINE.md` and `REFERENCE_CELL_MAP.md` — original-to-curated cell maps
- `funnybird_notebooks/` — original inspiration notebooks outside `curated`

## 3. Why RL was introduced

The standard target `c_j` is derived from the render parameters: a
`placeholder`/removed part already gives an all-zero concept group, but a
non-placeholder part can stay positive when its part map has almost no visible
pixels. RLv2 additionally uses visibility area:

`c_j^RL = c_j * visible_j`

The images, architecture, optimizer, loss family, and gamma are held fixed; the
concept targets change.

The corrected RLv2 builder:

- excludes zero-pixel cases from the visible-size median;
- computes medians on training data and reuses them for test;
- refuses missing part maps;
- keeps results separate under `*-rlv2`.

RLv2 changed 7,940 training images:

- tail 7,489
- beak 367
- eye 268
- foot 48
- wing 12

This intervention tests whether the visibility/label conflict explains part of
tail grounding failure. It is not expected to prove every mechanism.

## 4. Important evaluation correction

Legacy standard and RLv2 swap CSVs were produced by separate live-renderer runs.
Their structural jobs were intended to match, but `pixel_count_cf` differed across
runs. Therefore the old standard-versus-RLv2 graph is not an exact paired-image
causal comparison.

This does **not** erase the within-model non-RL discovery story. It means that
precise cross-model differences must be recomputed on identical image bytes.

Commit `7973864` adds the repair:

- shared cached counterfactual RGB images;
- serialized cache creation;
- saved `li`, `render_id`, paths, and SHA-256 hashes;
- fail-closed validation that every model uses the same render-ID set and hashes;
- new output directory `$CURATED_DATA/swap_fixed_v1`;
- notebook 03rl refuses legacy swap CSVs.

The repaired seed-1 evaluation reuses existing checkpoints. No retraining is
required for seed 1. Rendering happens once; all models replay the cache.

### 2026-07-29 semantic-renderer failure

Job `3329834` completed and loaded all six requested `epoch_100.pt` checkpoints,
but its saved examples proved the evaluation invalid:

- all 15 `orig`/`swap`/`delete` RGB examples were byte-identical, nearly black
  images with only 16 non-black pixels;
- all five part maps were byte-identical, with only four non-black pixels;
- the old validator passed because it checked one hash per render ID and
  cross-model agreement, not image content or diversity.

Therefore `$CURATED_DATA/swap_fixed_v1`, its render cache, its
`fixed_rl_comparison.csv`, and every plot derived from that comparison are
quarantined. In particular, apparent RLv2 tail gains of `+0.173` (CBM), `+0.119`
(MCBM gamma 0), and `+0.211` (MCBM gamma 0.1) are not causal evidence.

This does not prove that the older curated notebook 02/03 renderer runs had the
same failure. The executed notebook 02 HTML preserves a visibly non-degenerate
example grid and reports nonzero changed-pixel counts for every part
(tail 349, wing 827, beak 111, foot 734, eye 177). Those are useful spot checks,
not proof of every legacy render. Renderer-dependent legacy conclusions remain
provisional until regenerated with the new fail-closed semantic preflight.

The driver now requires, before model loading or cache reuse: deterministic RGB
and part-map output, a live render close to its stored FunnyBird reference, a
non-degenerate RGB image, visible RGB changes for every part swap and deletion,
and target-colour pixels in every tested part map. It saves
`renderer_preflight/renderer_semantic_preflight.png` and JSON. The validator also
requires render diversity, counterfactual-versus-original changes, part-map
diversity, and positive target-pixel counts. The next clean output is
`$CURATED_DATA/swap_fixed_v2`; do not reuse the v1 cache.

The driver also records the intervention-sensitive quantity
`response_delta = (z_donor-z_source)_swap - (z_donor-z_source)_orig` and
`swap_moved_toward_donor = response_delta > 0`. This must be primary for causal
interpretation. Absolute post-swap `ordering_correct` is secondary because the
v1 black-image run showed that coordinate priors alone can reproduce a plausible
tail-low/foot-high hierarchy.

## 5. Current training state

Known before the latest submissions:

- Standard CBM: trained seeds 1–3.
- Standard MCBM: gamma 0 and 0.1 have seeds 1–3; gamma 0.3, 1, 3, 5 have seed 1.
- Corrected CBM-RLv2: seed 1 trained and legacy swap completed.
- Corrected MCBM-RLv2: gamma 0 and 0.1 seed 1 trained and legacy swaps completed.
- Old `*-rl` models used the earlier flawed relabeling and must not be mixed with
  `*-rlv2`.

Submitted on 2026-07-24:

- job `3322224`: CBM-RLv2 seeds 2 and 3.
- job `3322225`: MCBM-RLv2 gamma 0 and 0.1, seeds 2 and 3.

These training jobs are useful but are not required to obtain the repaired seed-1
comparison.

Corrected RLv2 gamma 0.3, 1, 3, 5 are not required to answer whether the label
intervention helps tail at gamma 0/0.1. Train them only if making a claim about
RL across the full gamma sweep.

## 6. What is needed for a conclusive RL claim

Minimum claim: “visibility-aware labels reproducibly improve tail grounding.”

- [ ] **A. Fixed-image seed-1 comparison:** standard and RLv2 evaluated on identical
      `image_cf_sha256` values.
- [ ] **A0. Matched checkpoint:** every seed-1 comparator loads `epoch_100.pt`;
      standard CBM's later `epoch_150.pt` is not admissible against RLv2 models
      that end at epoch 100.
- [ ] **B0. Semantic renderer preflight passes:** stored and live reference agree;
      every part's saved `orig`/`swap`/`delete`/`part_map` row is visibly valid.
- [ ] **B. Fixed-image validation passes:** same render IDs and hashes in every
      compared CSV, with non-degenerate hash diversity and real
      counterfactual-versus-original changes.
- [ ] **C. Replication:** evaluate RLv2 seeds 2 and 3 on the same cache after jobs
      3322224/3322225 finish.
- [ ] **D. Seed-level consistency:** tail improvement has the same direction across
      seeds; report seed values, not row-level pseudo-error bars.
- [ ] **E. Independent deletion check:** compare standard versus RLv2 deletion on
      visibly present parts, excluding no-op deletions.

For a seed-1 provisional claim, A+B are enough if clearly labelled provisional.
For a stable general claim, A–E are needed.

Full corrected high-gamma RLv2 training is optional unless the paper claims the RL
effect across every gamma.

## 7. Notebook 03rl must be expanded after fixed results

It currently has too few figures. It should tell this discovery story:

1. Example of an original positive concept label with almost no visible part.
2. Equation and plain explanation of `c_j^RL`.
3. Label-change counts by part.
4. Ordinary accuracy/concept-accuracy sanity check.
5. Fixed-render audit: matching `li`, `render_id`, and SHA-256.
6. CBM all-part standard-versus-RLv2 comparison.
7. Per-render paired margin change:
   `delta_margin_i = margin_i^RL - margin_i^standard`.
8. Forward/backward split.
9. Visibility-stratified tail result on the fixed images.
10. Tail-variant result/confusion.
11. MCBM gamma 0 and 0.1 comparison; standard full-gamma context may be shown
    separately.
12. Other-part effects, without calling inconsistent one-seed changes benefits/costs.
13. Downstream donor-species probability.
14. Independent deletion, explicitly pending until available.

Do not duplicate every CBM figure merely to make 03rl long. Include the controls
needed to establish the intervention and link back to notebook 02 for discovery.

## 8. User’s notebook-review rules

These are mandatory:

1. Before interpreting a figure, display it in chat with a plain caption.
2. Explain axes, colors, reference lines, and variables (`z_source,cf`,
   `z_donor,cf`, `margin`, `gamma`, `c_j`).
3. Before every test, state the question, why the test answers it, and what each
   possible result would mean.
4. Write a natural first-reader progression: observe, ask why, test, then dig deeper.
5. Do not write backward from the desired final conclusion.
6. Separate “the graph directly shows,” “this supports,” and “still unproven.”
7. Investigate contradictions rather than averaging them away.
8. Name the original inspiration plot/cell; inspect `funnybird_notebooks/` outside
   `curated`.
9. After executed HTML is uploaded, reread every image as if no prior conclusion
   were known.
10. Use simple language but retain variable names.
11. Explain unusual scales, including CBM logits in the tens versus MCBM values near
    the +/-3 target.
12. Never use row-level error bars as independent uncertainty when rows reuse
    images/species. Show seeds.

## 9. CUB status

CUB and CUB70 loaders passed real-data smoke tests with 112 concepts and 28 groups.
CBM jobs submitted:

- `3322211`: full CUB CBM
- `3322212`: CUB70 CBM

Check their current state before assuming completion. Notebooks 05/06 require the
corresponding trained/evaluation outputs.

## 10. Rule for the next assistant

Do not issue a long cluster command until:

1. its inputs and expected outputs are checked in the current repository;
2. the exact Slurm output filename is derived from the script;
3. the command is limited to the shortest missing step;
4. success and failure signatures are stated;
5. existing checkpoints/results are reused whenever possible.
