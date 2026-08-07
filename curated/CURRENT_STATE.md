# Current experiment state

## 2026-08-06 notebook-03 ground-up presentation correction

Notebook 03 now follows the approved notebook-02 report contract rather than a
generic checklist. Before any result it states the inherited standard-CBM event,
the exact conditions required to call MCBM a repair, the complete figure ladder,
the three contributor hypotheses plus source-species residual, MCBM-specific
capabilities and limits, preregistered directions, one consolidated model/loss
diagram, a complete symbol table, and the dataset/population boundary.

MCBM-specific requirements are explicit: standard CBM versus MCBM gamma zero is
the architecture/noise baseline; gamma zero versus positive gamma is the
minimality test; compression of internal `h` is separated from the learned
`h -> z` head; counterfactual `h` is unavailable in the accepted swap CSVs; and
compression alone cannot count as grounding repair. A new
`--preserve-outputs` builder option synchronizes corrected Markdown into an
executed notebook without rerunning or altering numerical code outputs.

All important outputs—Figures 1, 2, 2b, 2c, 2d, 3, 4, 5, 6, 7, 7b, 8, 8b, 9,
9b, 10, 11, 11b, 12, 13, and 14—were displayed in chat in report order and
reviewed using question, variables/prediction, literal observation, alternative,
discriminating test, limited conclusion, and next question. No numerical result
changed during the presentation correction.

Final limited decision: gamma implements strong representation compression while
ordinary concept health remains high, but it is not a general grounding repair.
On the one accepted causal seed per gamma, tail donorward response falls from
`18.20` to `6.27`, controlled backwash rises from `0.506` to `0.666-0.780`, and
exact-value error rises from `0.722` to `0.798-0.891`. Wing and foot remain
strongly grounded; beak and eye improve at selected gammas. Identified
contributors do not reduce the held-out residual to zero, and all-gamma causal
seed replication remains `INCOMPLETE`.

## 2026-08-06 notebook-03 bounded follow-up reviewed

Notebook 03 has now been rebuilt to the same presentation contract as notebook
02. Every analysis section explicitly contains: its notebook-02 connection,
question, variables and predicted direction, method/exclusions, numbered figure,
complete axis/panel/color/denominator explanation, numerical sign example where
needed, and the post-figure evidence chain. The previously reviewed numerical
conclusions remain in the builder; the generated notebook is intentionally
unexecuted until it is rerun on Adroit.

The first execution of Figure 2c confirmed strong per-part `h` compression, but
its median-slope summary was ambiguous for a piecewise-linear ReLU head: a zero
median can mean a majority of locally flat held-out rows rather than a globally
constant head. The revised Figure 2c therefore reports mean `|dz/dh|`, the
locally flat-row fraction, and positive/negative/flat fractions explicitly.
Figure 2d repeats every tail gamma outcome after excluding all swaps involving
tail value 7, the sole exactly collapsed gamma-zero output. Both bounded tests
use saved predictions/checkpoints only and have now been executed and displayed.

Figure 2c shows that positive gamma compresses every part, but compensation by
the learned head is not uniform. Tail has the largest within-label `h` spread at
every positive gamma, and its mean local `h -> z` sensitivity falls from `1.86`
at gamma `0.1` to `0.48` at gamma `5`; wing and foot remain near `1.6` at gamma
`5`. This is **ACCEPTED FOR LOCAL HEAD BEHAVIOR**, not as causal localization of
the counterfactual response, because the fixed-render CSVs do not contain
counterfactual `h`.

Figure 2d removes every source/donor value-7 tail swap and retains 930 rows per
gamma. The all-row and exclusion curves nearly overlap: excluded-row backwash is
`0.523` at gamma zero and `0.665-0.780` at positive gamma, while donorward
response falls from `18.73` to `6.39`. This is **ACCEPTED FOR COLLAPSE
SENSITIVITY**: the one collapsed output does not create the tail gamma result.

## 2026-08-06 standard-MCBM visual review

The corrected standard-MCBM notebook 03 at commit `7f25843` executed on Adroit.
All 19 displayed outputs (Figures 1, 2, 2b, 3, 4, 5, 6, 7, 7b, 8, 8b, 9, 9b,
10, 11, 11b, 12, 13, and 14) were displayed in chat and reviewed against their
printed numerical tables. The review records are now stored in
`analysis/build_03_standard_mcbm_report.py`.

Accepted limited result: gamma strongly compresses the intended MCBM internal
slots while ordinary species and concept accuracy remain stable. On the one
validated fixed-render seed, this compression is not a general grounding repair.
Tail donorward response falls from `18.20` at gamma zero to `6.27` at gamma five,
tail controlled backwash rises from `0.506` to `0.666-0.780` at positive gamma,
and exact tail-value error rises from `0.722` to `0.798-0.891`. Wing and foot
remain strongly grounded, while beak and eye improve at selected higher gammas.
This is an all-gamma seed-1 causal result; fixed-render training-seed replication
remains `INCOMPLETE`.

The proposed contributors do not sum to a complete explanation. Visibility,
exact value, and source species each lower held-out final-margin prediction error,
but `0.50-0.67` standardized RMSE remains after all are included. Training
label/visibility conflict strongly matches the tail-versus-wing/foot ordering but
is shared across gamma and therefore cannot explain the worsening tail gamma
curve. Species decoding and recall remain supporting diagnostics, not substitutes
for the controlled swap.

Figure 2b's corrected full-range test is now executed. Only `gamma=0, tail_7`
is exactly collapsed. The gamma `1`, `3`, and `5` central ties contain 184-211
distinct finite values and full ranges of 12.32-16.74, so they are not called
collapsed. Figure 2d tests whether excluding the one collapsed value changes the
tail result before that result is accepted without qualification.

## 2026-08-05 CBM-only RLv2 report split

The causal label test is now separated by model family. Notebook
`02rl_funnybirds_cbm_relabeled.ipynb` compares only standard CBM with matched
CBM-RLv2. It executed successfully at seed 1 on 5,000 identical fixed-render
swaps; all 15 new figures were displayed and reviewed in chat. Notebook 03
remains standard MCBM. Notebook 03rl remains the later MCBM-RLv2 gamma extension
and must not be used as the primary CBM causal proof. The exact build contract
is `NOTEBOOK_02RL_ROADMAP.md`; the runner is
`notebooks/run_02rl_notebook.sh`.

The seed-1 controlled candidate rate fell most for tail (0.608 to 0.440), then
beak (0.537 to 0.417) and eye (0.491 to 0.429). Foot was nearly unchanged
(0.084 to 0.078), while wing worsened (0.133 to 0.156). Tail followed the
preregistered raw-score mechanism: donor score +2.325, old-source score -3.665,
and final margin +5.991 raw-logit units. Many events remain, and new events were
introduced. Direction and visible-pixel checks support the tail/beak result;
exact donor-value recognition improves but remains unequal. Source-species
residuals remain descriptively, but adding source species worsened held-out
margin prediction, so it is not accepted as a generalizing explanation here.
The bounded conclusion is a provisional causal effect of the complete matched
RLv2 label intervention at seed 1. CBM-RLv2 seeds 2-3 remain `INCOMPLETE` for
training-seed reproducibility.

## 2026-08-03 standard-CBM report rebuild

First execution of the rebuilt pair stopped before Figure 1 in notebook 02 with
`max |z-logit(p)|=46.95`. This was a report-definition error, not a failed model:
both standard CBM configs use learned `1 -> 3 -> 1` concept heads. Saved
prediction field `z` is the encoder latent slot, while the fixed-render swap
driver correctly records post-head `c_logits`. `analysis/minimal_cbm_scores.py`
now replays the saved concept heads on saved latent slots and checks that their
sigmoid exactly reproduces saved `c_preds`. `cub70_export_eval.py` now uses the
same replay, because its earlier parquet files also mislabeled latent `z` as the
raw concept logit. The cross-dataset runner refreshes both CUB exports before
executing the reports. No retraining or Slurm job is needed for this correction.
The first corrected-export attempt then stopped with `ERROR` before notebook
execution because the exporter copied the prediction suffix `.pth` when looking
for a model checkpoint; minimal_cbm model checkpoints use `.pt`. The lookup now
maps `predictions/epoch_100.pth` to `models/epoch_100.pt`. No scientific output
was produced by that interrupted attempt.
The next execution completed notebook 02 and reached notebook 05 Figure 2, then
stopped with `ERROR` because exact CUB concepts without a released-mask mapping
produced missing Matplotlib colors. The label-only table remained valid; no
Figure 2 interpretation was accepted. The plot now uses explicit gray for those
concepts and labels gray as `no released-mask mapping`.
The following execution reached notebook 05 Figure 4 and stopped with `ERROR`:
adding `mask_group` to the Figure 2 support table caused the Figure 4 merge to
rename duplicate columns to `mask_group_x/y`. Figure 4 had no scientific output.
The merge now drops the plotting-only duplicate and retains the row-level
`mask_group`. The finalizer also now raises `ERROR` when any expected figure cell
has no PNG; it can no longer report a metadata pass for an interrupted notebook.
The next notebook-05 execution reached Figure 8 and stopped with `ERROR` because
the matching code used processed CBM `gt_label`. Those labels are majority-voted
and effectively species-constant, so no species had the required within-species
positive and negative support. This was the wrong input for the restored CUB
recall diagnostic. Figure 8 now joins original per-image
`image_attribute_labels.txt` annotations to raw-`z` predictions using explicit
image and attribute IDs, reports alignment coverage, and applies the vectorized
CUB matching rule from `mcbm_recallv4`. `cub70_export_eval.py` now records the
original CUB `attribute_id` needed for that exact join.

Notebook sources 02 and 05 have been rebuilt from the ground up by
`analysis/build_standard_cbm_reports.py`. The new main reports contain 13
FunnyBird plot cells and 14 CUB70 plot cells, use raw concept logits for
grounding, define the implemented CBM before using its variables, and move the
uncalibrated deletion/patch/paste methods to appendices. They contain no MCBM or
RL result in the discovery chain.

Both rebuilt reports executed successfully and synchronized HTML exports were
pushed in commit `6e08d0c`. All 28 numbered analysis sections (including 2/2b,
6/6b, 7/7b, 8/8b, and 12/12b) and the separate FunnyBird renderer preflight were
displayed in chat and reviewed on 2026-08-04. The review records and evidence
ledgers now state the accepted, negative, and limited results explicitly. No
Slurm job was used. Acceptance checks completed:

1. both saved concept-head replay/raw-logit checks pass;
2. both notebooks execute without an exception;
3. `finalize_standard_cbm_reports.py` reports described PNG outputs;
4. HTML exports are created;
5. every numbered figure was displayed in chat before its placeholder review was
   replaced.

The accepted standard-CBM result is asymmetric across datasets. FunnyBird gives
the causal predicate: all five controlled replacements move raw logits toward
the donor, while the old source still wins most often for tail, beak, and eye.
Visibility contributes, and training-label conflict is largest for tail, but the
identified contributors do not sum to zero residual. The held-out accounting
test improves only after visibility; exact values and source species worsen its
prediction error and therefore receive no generalizing explanatory credit from
that test. CUB gives converging observational evidence rather than an equivalent
swap: released-mask-absent context gaps are positive for 48 of 50 eligible exact
concepts, raw-label-matched species gaps remain, and species lowers held-out
raw-z prediction error after exact concept and mask state. However, mask absence
often means missing/coarse annotation rather than physical occlusion, natural
visibility effects have mixed signs, and two of 112 outputs are exactly
collapsed. CUB therefore supports contextual organization but not a causal
donor/source backwash event.

This rebuild changes presentation and analysis definitions; it does not by itself
upgrade or invalidate any prior scientific claim.

Predicate and final-claim status is centralized in
`PREDICATE_PROOF_LEDGER.md`. Its same-row residual accounting is mandatory:
visibility filtering is selection, matched RLv2 is a causal label intervention,
and variant/source-species residuals are observational until independently
manipulated. CUB currently stops before the causal waterfall because its edit and
positive-donor-response predicates failed.

The primary cross-dataset comparison is standard CBM notebook 02 versus standard
CBM notebook 05, mapped question by question in
`CBM_CROSS_DATASET_PROOF_MAP.md`. Notebook 03 (MCBM) and notebook 03rl (RLv2)
remain later follow-ups and must not replace that discovery comparison.

Last repository reconciliation: **2026-08-02**, through executed complete CUB
approximation-chain commit `16adb39` plus the unexecuted beak/tail insertion
pilot described below.
Last live Slurm observation: job `3333238` (`fb_rl_broad_s1r`) was running at
1:54:52 on **2026-07-31**. Live cluster state can change and must be refreshed
with `squeue -u "$USER"` before the next cluster decision.

The reciprocal FunnyBird/CUB70 CBM deletion suite was executed on **2026-08-02**.
Its computation completed, but the preregistered FunnyBird calibration **FAILED**;
the CUB70 causal interpretation is quarantined. This run submits no Slurm job and
does not overturn the clean FunnyBird renderer evidence.

Notebook 05 now contains a plain-language failed-test account with concrete CUB70
beak and FunnyBird eye/tail examples. It explains the large artificial fill, the
meaningful-part control damage, the missing FunnyBird wing control, the 148/2,500
selection loss, and why the surviving species association remains observational.

Update this file after any submission, completion, failure, validation,
cancellation, or repository correction. “Submitted” never means “proved.”

## Research order and present decision

| Order | Stage | Present state | Next proof step |
|---:|---|---|---|
| 1 | Non-RL FunnyBirds data | Static dataset claims accepted | No rerun |
| 2 | CBM discovery | Existing FunnyBird results retained; reciprocal deletion executed and calibration failed | Keep clean renderer evidence; use shared run only as a documented failed discriminating test |
| 3 | MCBM minimality | Compression/deletion accepted with gamma-saturation caveat; legacy swaps provisional | Finish fixed-cache standard gamma replay and inspect it |
| 4 | RL causal follow-up | Core seed-1 fixed-render notebook executed and 20 figures inspected; broad γ replay running | Finish `3333238`, rebuild all-γ notebook, then fixed-render seeds 2–3 |
| 5 | CUB/CUB70 | CBM approximation chain executed; whole-part, small-patch, and beak/tail insertion tests failed their scientific gates | Finish notebook 05 with the causal boundary, then let notebook 06 test only how minimality changes the observational CUB behavior |

## Randomized small-mask dose response (FunnyBird executed; calibration failed)

The attempted CUB70 proof step is implemented in
`analysis/randomized_patch_masking.py`,
`analysis/compare_randomized_patch_masking.py`, and
`analysis/run_randomized_patch_masking.sh`. It submits no Slurm job and must run
inside an already allocated GPU session.

The driver is fail-closed: it runs FunnyBird first and exits before CUB70 unless
all preregistered calibration checks pass. The test uses four increasing partial
masking doses, four random placements, Gaussian-soft patch edges, local-blur and
local-mean fills, and exact translated target/other-bird/background masks. It stores raw `z`,
probability, every component response, mask dose, image hashes, and visual sheets.

Acceptance signatures:

- `[RANDOMIZED PATCH MASKING PASS]` for the FunnyBird computation;
- `[FUNNYBIRD PATCH CALIBRATION PASS]` before any CUB70 computation starts;
- `[RANDOMIZED PATCH MASKING PASS]` for CUB70;
- `[CROSS-DATASET PATCH ANALYSIS PASS]` and a `PASS` audit JSON.

The maximum allowed conclusion is robust local pixel reliance plus partial
contextual retention. This is not a renderer-quality swap and cannot by itself
identify species as the retained source. Identical pending sections were added
to notebooks 02 and 05; execute them only after the suite passes, then display
every generated figure and intervention sheet in chat before interpretation.

First FunnyBird execution reached 500/500 image-parts in 5m25s but stopped at
the RGB-change gate because at least one target/other-bird edit was a no-op.
This can occur when local blur is applied to an almost uniform rendered surface.
The driver now preserves pre-gate rows, drops the complete matched
target/other-bird/background unit whenever either required edit changes no RGB,
records no-op counts by fill/location, and fails if filtering removes any
part/fill or leaves fewer than three doses. The completed inference from the
failed process was not written by the old version, so the five-minute FunnyBird
stage must be rerun; CUB70 correctly never started.

The corrected retry completed 500/500 image-parts in 5m28s and wrote 14,133
post-gate rows, then returned `[FUNNYBIRD PATCH CALIBRATION FAIL]`. CUB70 again
correctly did not start. This is now a completed failed discriminating test, not
a software crash. Do not relax the preregistered gate. Inspect the calibration
audit, table, figure, and every FunnyBird mask sheet before deciding which
assumption failed. Notebook 02/05 patch sections now load and display a failed
FunnyBird-only calibration without requiring nonexistent CUB70 outputs.

The supplied review archive was inspected completely in chat: all three summary
figures and all eight saved intervention sheets (two each for beak, eye, foot,
and tail) were displayed. No wing sheet or wing row exists. On the probability
scale, nearly all small-patch effects round to zero because the CBM outputs are
saturated near one. On the standardized raw-`z` scale, target masks move downward
with dose for every surviving part under both fills while other-bird/background
controls stay near zero: beak is strongest and tail is weakest. This supports
local-pixel use only in the selected surviving examples. It does not rank wing
against tail, establish population-wide effects, identify species as the retained
source, or license CUB70.

The saved input audit now explains wing exactly: 497 wing image-parts were
eligible and 100 were selected, but wing produced no raw pre-gate rows and no
post-gate coverage. Therefore all selected wing repeats failed at control-mask
construction. The rigid method attempted to translate the complete wide wing
patch union as one shape onto non-wing support. The 65 later no-op losses were
all local-mean other-bird edits and were not the cause of missing wing.

A post-hoc v2 is implemented in `analysis/run_randomized_patch_masking_v2.sh`.
It preserves the failed v1 outputs, places each Gaussian control patch
independently while matching patch count, sigma, and total alpha mass, and uses
standardized raw `z` rather than saturated probability as the primary calibration
metric. It remains fail-closed: all five FunnyBird parts, grounded wing/foot
controls, two-fill agreement, and dose direction must pass before CUB70 starts.
Expected output root: `$CURATED_DATA/randomized_patch_masking_v2`. Acceptance
signatures are `[FUNNYBIRD V2 GATE PASSED]` followed eventually by
`[RANDOMIZED PATCH V2 SUITE COMPLETE]`; any calibration FAIL stops before CUB70.
Use `STOP_AFTER_FB=1` for the shortest calibration-only run. After that passes,
`REUSE_FB=1` rechecks the saved gate and proceeds to CUB70 without recomputing
the FunnyBird forward passes.

The first v2 execution completed all 500 selected image-parts in 6m13s and wrote
34,107 matched rows, so the corrected placement recovered substantially more
coverage, including wing pending audit confirmation. The subsequent comparison
crashed before calibration because the single-value pandas pivot named its
columns `target`/`other_bird`/`background`, while the new raw-`z` code expected
prefixed names. This was a reporting-code bug, not a failed model result. Commit
after this entry renames those columns explicitly. Reuse the saved parquet with
`REUSE_FB=1 STOP_AFTER_FB=1`; do not repeat the forward passes.

After the pivot fix, the saved 34,107-row output was compared twice with
`REUSE_FB=1 STOP_AFTER_FB=1`. Both checks returned
`[FUNNYBIRD PATCH CALIBRATION FAIL]`; CUB70 correctly did not start. This is now
a genuine failed v2 gate, not an inference or reporting crash. Do not rerun the
saved comparison again and do not weaken a check before inspecting the v2 audit,
calibration table, all summary figures, and every saved intervention sheet.

The complete v2 archive was inspected on 2026-08-02. Five of the six registered
checks passed. The sole failure was agreement between the two fill methods on the
ordering of parts (`Spearman = 0.0`). This disagreement is localized to wing:

- with blur, wing had the largest target-specific standardized raw-`z` drop
  (`2.037`);
- with mean fill, the target edit itself still had a large drop (`2.193`), but the
  other-bird control also had a very large drop (`1.415`), leaving an adjusted
  drop of only `0.095`;
- the saved sheets show that wing controls scatter patches over several meaningful
  non-wing bird regions;
- the current `local_mean` implementation assigns one median colour from a single
  bounding box around all scattered patches. It is therefore not local per patch
  and can make the scattered control unusually destructive.

The target-only result is consistent across fills: tail is weakest, wing is
strong, and beak is also strong. This supports local pixel reliance in the selected
examples, but it does not by itself measure backwash. Beak is the important
counterexample: a part can produce a large local masking response while the donor
swap still fails because the old source/context response remains stronger.

Coverage also remains inadequate for wing. Although 100 wing image-parts were
selected, 368 lacked enough non-wing support for the matched control; only 11 wing
images (32 highest-dose matched rows per fill) entered calibration. Therefore the
v2 wing estimate is not population-representative. This blocks transferring the
**patch intervention** to CUB70; it does not invalidate CUB70's separate natural-
visibility, label-conflict, species-probe, exact-concept, or species-matched
approximations. Preserve the failure and move through those comparisons. A lower-
dose/local-fill calibration is optional if the patch claim is retained; it is no
longer the main gate for the complete CUB notebook. A calibrated 2-D swap is the
final stronger follow-up, not the next prerequisite.

## What the FunnyBird analyses did not explain away

The FunnyBird result was never reduced to zero. Standard CBM tail replacement
ordering was about 0.35--0.37 (roughly 63% violations), and visible-only filtering
helped only modestly. Visibility-aware RLv2 reduced tail backwash in the matched
seed-1 comparison but left the mean margin slightly negative, ordering below 0.5,
and strong variant/source-species differences. Thus the current evidence supports:

- test-time visibility/occlusion as a partial cause, especially for beak/eye and
  modestly for tail;
- visibility-conflicting training labels as one causal contributor to tail;
- exact visual-variant difficulty as an additional observed contributor;
- an unchanged body/source-species association remaining after variant adjustment,
  still observational because body context was not independently manipulated.

These mechanisms overlap and were not expected to add up mechanically to zero.
The paper claim must be "multiple contributing mechanisms with a residual," not
"all causes identified." CUB should test whether the same candidates receive
converging support, while stating where its approximations are weaker.

## CUB70 visible beak/tail insertion pilot (executed; scientific gate failed)

This is the shortest remaining CUB CBM discriminating test. It does not submit a
Slurm job and must run inside an already allocated GPU session. For each clearly
visible beak or tail target, it makes four versions of the same photograph:
original, deleted target, a same-value part paste, and a different-value donor
paste. The primary metric is the change in `donor z - source z` between the
different-value paste and the same-value paste.

Implementation:

- `analysis/cub70_beak_tail_swap_pilot.py`
- `analysis/run_cub70_beak_tail_swap_pilot.sh`
- notebook-05 pending section installed by
  `analysis/add_05_cub70_beak_tail_pilot.py`

Expected output root:
`$CURATED_DATA/cub70_beak_tail_swap_pilot/`.

Computation signature:
`[CUB70 BEAK/TAIL PILOT COMPUTATION PASS; VISUAL REVIEW REQUIRED]`.
The run wrote 80 pairs: 40 beak and 40 tail. All sixteen saved sheets were
displayed and inspected in chat. Most masks hit the intended region, but beak
edits were often only a few pixels, several tail pastes looked like flat texture
strips, and one hummingbird beak edit was visibly distorted.

The preregistered scientific gate failed:

- median donor response versus the same-value paste was `-0.0037` for beak and
  `0.0000` for tail;
- only 40% of pairs moved in the predicted positive direction for either part;
- deleting the named part lowered its source score in only 47.5% of cases for
  either part;
- final median donor-minus-source margins were negative, but these cannot be
  interpreted as retained-source backwash because the donor-response prerequisite
  failed first.

Do not tune or expand this pilot. It supplies no reliable within-image CUB donor
response. Preserve it as a negative discriminating test. The CUB conclusion is
now observational ingredients consistent with possible backwash, not causal
backwash at FunnyBird strength. This does not change the validated FunnyBird
renderer result.

## Reciprocal FunnyBird/CUB70 CBM test (executed, calibration failed)

The suite ran inside an allocated GPU session. It does not need to be rerun.

| Item | FunnyBird | CUB70 | Acceptance signature/status |
|---|---|---|---|
| Exact positive concept | all five rendered parts | every selected concept with an available mapped mask | implemented; pending execution |
| Target visibility gate | exact part map, at least 0.1% of image | released combined coarse mask, same threshold | implemented; exclusion counts saved |
| Equal-damage control | translated identical mask, nonoverlapping, at least 70% on bird | identical rule | implemented; pairs without a valid control are skipped and counted |
| Four raw-z inputs | original, target-deleted, control-deleted, part-only | identical | executed; every component retained |
| Calibration | shared deletion versus epoch-100 clean renderer deletion | CUB interpretation depends on this | **FAILED**: eye/tail near-zero shared drop, wing absent; quarantine |
| Shared plots | reciprocal section in notebook 02 | identical reciprocal section in notebook 05 | inspected; notebook reasoning updated, re-execution pending |
| Species residual | bias-corrected source-species effect after fixing exact concept, with 200 label permutations | identical | observational only; never sufficient alone |
| Crude copy-paste swap | calibration target is the clean renderer swap | quarantined | secondary and not yet implemented/accepted |

Expected outputs under `$CURATED_DATA/paired_deletion/`:

- `funnybirds-clean-renderer-epoch100-s1.parquet`
- `funnybirds-cbm-s1.parquet` plus audit, summary, and example sheets
- `cub70-cbm-s1.parquet` plus audit, summary, and example sheets
- `comparison/paired_deletion_audit.json`
- `comparison/funnybird_deletion_calibration.png`
- `comparison/paired_deletion_main.png`
- `comparison/paired_deletion_species.png`

Observed counts: FunnyBird evaluated 148/2,500 image-parts and omitted wing;
CUB70 evaluated 7,111/13,498. All three summary figures and all 24 supplied
intervention sheets were displayed and inspected in chat. Large inpainting blobs,
meaningful-part controls, and severe part-only distribution shift were visible.
The broad CUB70 species residual is a hypothesis-generating association only.

Work proceeds in parallel. Finish carrying the original standard FunnyBirds work
into curated notebooks 01–03 while matched RLv2 jobs run. As soon as the matched
seed-1 fixed-cache evaluation finishes, build a clearly labelled **seed-1
provisional** RL result for the professor; do not wait for seeds 2/3. Seeds 2/3
then test whether that result is stable and may revise it. CUB preparation and
completed-output inventory can also proceed meanwhile, while the written story
still keeps RL after the standard CBM/MCBM evidence.

## Notebook synchronization

| Notebook | Source state | Executed-output state | Decision |
|---|---|---|---|
| `01_funnybirds_analysis` | Present | Executed | Keep first: non-RL data and species/concept structure |
| `02_funnybirds_cbm` | Existing discovery chain plus reciprocal failed-test reasoning | Existing 18 figures inspected; shared artifacts inspected in chat; updated cells need execution | Preserve clean renderer conclusion and document why shared deletion failed |
| `03_funnybirds_mcbm` | Full standard-MCBM explanation chain including all-gamma variant confusion | 23 figures inspected; no execution errors | Tail exact-variant attribution does not improve with gamma in seed 1; other parts are mixed; replication pending |
| `03rl_funnybirds_mcbm_relabeled` | Core plus dynamic all-γ and paired-point diagnostics | Core γ=0/0.1 execution inspected; rerun after `3333238` | RL causal follow-up only |
| `04_cub_analysis` | Revised with explicit FunnyBird-data mapping and CUB limits | Executed and exported in `8d65c97`; 3 figures inspected | CUB data stage |
| `05_cub_cbm` | CUB70-CBM exploration plus identical reciprocal failed-test reasoning; no MCBM or relabeling | Existing 19 figures plus all shared artifacts inspected; updated cells need execution. Mask-coverage loss remains accounted for. | Retain species residual as observational; run calibrated patch robustness next |
| `06_cub_mcbm` | Revised with direct γ mapping and collapse/task guards | Not executed; MCBM exports pending | CUB minimality stage after CBM questions are fixed |

Do not treat a stale HTML as synchronized merely because it exists. When notebook
source changes, execute on Adroit, export HTML, inspect every important figure,
then commit the `.ipynb` and `.html` together.

## Evidence already accepted

The executed notebook confirmed that the archive has 67 class directories and
omits 1-based IDs `11`, `16`, and `32`. They account for 81 missing images:
30 Rusty Blackbird, 28 Painted Bunting, and 23 Mangrove Cuckoo. Seven additional
individual mask omissions remain: one Chuck-will's-widow, two Brandt Cormorant,
and four American Goldfinch photographs. The executed fuzzy filename audit
found no normalized archive match for any of those seven, so the
1,976-to-1,888 coverage loss is fully accounted for as archive omissions rather
than a failed filename join.

Exact CUB70-CBM collapse guard: `has_throat_color::grey` is constant;
`has_wing_pattern::multi-colored` is effectively constant; and
`has_throat_color::buff` takes only two rounded probability values. Do not use
these three slots as grounding evidence.

| Item | Evidence | Status |
|---|---|---|
| RLv2 versus standard record identity | `audit_03rl_accuracy.py` passed after constructing `funnybirds_processed_rl_trainval`; all record fields match except `attribute_label` | **DATA PARITY ACCEPTED** |
| Semantic renderer gate | Job `3330289`, exit 0; deterministic reference and visible swap/delete changes for every part | **ACCEPTED INFRASTRUCTURE** |
| Fixed-cache byte/hash validation | Job `3330701`, exit 0; six CSVs, 5,000 counterfactual RGB IDs, 250 original RGB IDs, hashes agree | **ACCEPTED INFRASTRUCTURE** |
| Behavior from job `3330701` | Models used different training populations | **QUARANTINED AS CAUSAL EVIDENCE** |
| Fixed-cache v1 | Nearly black, identical renders | **QUARANTINED** |

## Training and evaluation matrix

Legend: `yes` completed/available, `Q` exists but is quarantined for causal
comparison, `PD` submitted and last seen pending, `--` not submitted/required.

| Model | gamma | Standard s1 | Standard s2 | Standard s3 | Old RLv2 | Matched s1 | Matched s2 | Matched s3 | Matched fixed-cache evaluation |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CBM | -- | yes | yes | yes | Q s1 | PD `3331297` | PD `3331298` | PD `3331299` | PD `3331310` after seed-1 dependencies |
| MCBM | 0 | yes | yes | yes | Q s1 | PD `3331300` | PD `3331301` | PD `3331302` | PD `3331310` after seed-1 dependencies |
| MCBM | 0.1 | yes | yes | yes | Q s1 | PD `3331303` | PD `3331304` | PD `3331305` | PD `3331310` after seed-1 dependencies |
| MCBM | 0.3 | yes | -- | -- | -- | PD `3331306` | -- | -- | not in core matched evaluation |
| MCBM | 1 | yes | -- | -- | -- | PD `3331307` | -- | -- | not in core matched evaluation |
| MCBM | 3 | yes | -- | -- | -- | PD `3331308` | -- | -- | not in core matched evaluation |
| MCBM | 5 | yes | -- | -- | -- | PD `3331309` | -- | -- | not in core matched evaluation |

Corrected jobs use new result names containing `rlv2matched`; they must not
overwrite or resume old `rlv2` checkpoints.

## Full Slurm ledger

| Job(s) | Purpose | Last observed state | Current decision |
|---|---|---|---|
| `3331297` -> `3331298` -> `3331299` | Matched CBM seeds 1, 2, 3 | seed 1 `PD Priority`; later seeds `PD Dependency` | Keep; require parity signatures in every log |
| `3331300` -> `3331301` -> `3331302` | Matched MCBM gamma 0 seeds 1, 2, 3 | seed 1 `PD Priority`; later seeds `PD Dependency` | Keep; require parity signatures |
| `3331303` -> `3331304` -> `3331305` | Matched MCBM gamma 0.1 seeds 1, 2, 3 | seed 1 `PD Priority`; later seeds `PD Dependency` | Keep; require parity signatures |
| `3331306` | Matched MCBM gamma 0.3 seed 1 | `PD Priority` | Breadth run; not on core causal critical path |
| `3331307` | Matched MCBM gamma 1 seed 1 | `PD Priority` | Breadth run |
| `3331308` | Matched MCBM gamma 3 seed 1 | `PD Priority` | Breadth run |
| `3331309` | Matched MCBM gamma 5 seed 1 | `PD Priority` | Breadth run |
| `3331310` | Matched seed-1 fixed-cache evaluation | `PD Dependency` | Core evaluation; must use accepted v2 cache |
| `3322224` | Old CBM-RLv2 seeds 2/3 | Cancelled 2026-07-30 without running | Superseded by `3331298`/`3331299` |
| `3322225` | Old MCBM-RLv2 gamma 0/0.1 seeds 2/3 | Cancelled 2026-07-30 without running | Superseded by `3331301`/`3331302`/`3331304`/`3331305` |
| `3322220`-`3322223` | Old higher-gamma RLv2 jobs | Cancelled 2026-07-30 without running | Superseded by `3331306`-`3331309` |
| `3331180` | Open OnDemand Jupyter Generic session | Last seen running | Not training; close through Open OnDemand when unused |
| `3331248` | Standard high-gamma fixed-cache replay | Completed in 22m17s, exit 0 | Inspect its completion signatures and results; do not rerun |
| `3330289` | Semantic renderer gate | Completed, exit 0 | Accepted |
| `3330701` | Fixed-cache v2 old-model replay | Completed, exit 0 | Cache accepted; behavior quarantined |
| `3329834` | Fixed-cache v1 replay | Completed, exit 0 | Quarantined |
| `3322211` | Earlier MCBM/CUB sweep | Failed | Inventory outputs before any rerun |
| `3322212` | Earlier MCBM/CUB sweep | Completed | Inventory outputs before any rerun |
| `3322015`, `3322016` | Earlier CUB/CUB70 attempts | Failed | Inventory before rerun |

The six old held jobs are superseded submissions. They could run obsolete
payloads if released. Preserve their submitted scripts if historical provenance
is needed, then cancel them explicitly. Do not release them.

## Why the post-submission pull can still be safe

The corrected jobs were submitted while commit `94162f6` was checked out. Slurm
copies `train/sbatch_mcbm.slurm` at submission. That copied wrapper activates the
environment and calls one of these shared-filesystem paths at job start:

- `train/cbm_funnybirds_rl.sh`
- `train/mcbm_funnybirds_rl.sh`

The user pulled commit `30677c6` while every corrected training job was still
pending. A job starting afterward should therefore call the new scripts, which
derive matched configs from exact standard configs. This is not accepted on
reasoning alone. Every log must contain:

- record-identity checks and final `AUDIT PASSED`
- `[CONFIG PARITY PASS] ... only data.pkls_dir changed`
- a result name containing `rlv2matched`
- the intended gamma and seed

Absence of any signature means stop interpretation and audit the submitted
script/log.

## Next live reconciliation command

Run from `curated/`; it intentionally prints every user job first:

```bash
echo "===== ALL CURRENT USER JOBS ====="
squeue -u "$USER" \
  -o "%.10i %.28j %.2t %.10M %.24R"

echo "===== KNOWN EXPERIMENT ACCOUNTING ====="
sacct -j 3322220,3322221,3322222,3322223,3322224,3322225,3331180,3331248,3331297,3331298,3331299,3331300,3331301,3331302,3331303,3331304,3331305,3331306,3331307,3331308,3331309,3331310 \
  -X --format=JobID,JobName%28,State,Elapsed,ExitCode,End
```

Do not infer `COMPLETED` from a job disappearing from `squeue`.

## Acceptance order for matched runs

1. Data audit passes.
2. Exact config-copy audit passes; only `data.pkls_dir` differs.
3. Fresh checkpoint exists under the intended `rlv2matched` result name.
4. Ordinary accuracy is evaluated on the same validation identities.
5. Seed-1 models replay the already accepted fixed cache.
6. CSV hashes and render IDs match.
7. Notebook figures are read literally, one at a time.
8. If seed 1 is surprising, investigate before broad claims.
9. Seeds 2/3 determine reproducibility.
10. Higher gamma values support only the broader gamma question.
## 2026-08-05: standard MCBM paired-report rebuild

- Added `NOTEBOOK_03_06_MCBM_ROADMAP.md`, which locks the common FunnyBird/CUB
  questions and the valid dataset-specific substitutions before notebook 06 is
  interpreted.
- Rebuilt `notebooks/03_funnybirds_mcbm.ipynb` from the standard-CBM notation and
  controlled-backwash predicate. It covers every available gamma (`0, 0.1, 0.3,
  1, 3, 5`) and explicitly separates all-seed health from seed-1 fixed-render
  causal comparisons.
- Corrected the representation notation: prediction-file `z` is the internal
  slot `h`; raw concept logits are recovered through the saved learned concept
  heads and validated against saved probabilities.
- Added `notebooks/run_03_standard_mcbm_report.sh`. It uses existing checkpoints
  and validated renders only and submits no Slurm jobs.
- Notebook 03 is now **INCOMPLETE pending execution and figure-by-figure visual
  review**. Notebook 06 remains intentionally unchanged until the 03 questions
  and literal observations are accepted.

## 2026-08-06: notebook 03 earlier review superseded by direct-comparison rebuild

- The earlier 14-output version was reviewed, but it was not yet as diligent or
  directly comparable to notebooks 02/05 as required. Its accepted fixed-render
  observations remain provenance; the rebuilt report now requires a fresh
  complete visual review.
- The first review found and corrected three report defects: an unfinished
  `gamma=5, seed=2` artifact is now labelled **INVALID OUTPUT** and excluded,
  source-species residuals now include a scale-standardized panel, and the
  downstream class plot is separated by part.
- Accepted limited seed-1 result: gamma strongly compresses the intended
  representation while ordinary concept health remains stable, but it is not a
  general grounding repair. Tail donorward response weakens, tail final margins
  stay negative, and exact tail-value recognition worsens. Wing/foot remain
  strong; beak/eye improve at selected higher gamma settings.
- Every gamma still has only one validated fixed-render seed. Numerical gamma
  differences remain provisional pending independent fixed-render replication.
- The first rebuilt execution reached Figure 11 and stopped with `ERROR` because
  its recall cell incorrectly jumped directly to the all-positive-species
  fallback. A direct check established prediction/pickle identity and showed
  that the current curated validation labels vary within species (maximum exact-
  concept prevalence `0.87`; zero all-positive species pairs). Figure 11 now
  reproduces `fb_recallv2`'s actual two-stage rule: matched positive/negative
  species first, all-positive fallback only if applicable, with the selected
  rule and coverage printed. Figures 1--10 executed in the failed process but
  were not saved by `nbconvert --inplace`; no scientific interpretation was
  accepted from that interrupted run.
- It also adds exact-concept health, held-out sequential contributor accounting,
  known-label/raw-logit/internal-slot species controls, all-gamma exact-value
  support, and a direct four-measurement alignment with standard CBM.
- Status is **INCOMPLETE pending execution and figure-by-figure visual review**.
  Notebook 06 remains paused until that review is complete.
- Added `analysis/inventory_06_cub_mcbm.py` as the first notebook-06 step. It
  scans existing CUB/CUB70 MCBM checkpoints and predictions, rejects non-finite
  artifacts, identifies a matching epoch, and prints only the missing normalized
  export commands. It performs no training and submits no Slurm work.
## 2026-08-06: MCBM starting-margin/response decomposition added

- Corrected the report's boundedness explanation: MCBM penalizes internal `h`
  toward `-3/+3`, but neither `h` nor the learned post-head raw logit `z` is
  hard-bounded. Only `sigmoid(z)` lies in `[0,1]`.
- Added pending Figure 4b, which includes standard CBM and every MCBM gamma and
  decomposes the final margin exactly into starting margin, donor-score gain,
  and old-source-score decrease.
- Added pending Figure 5b, which separates all swaps into donor wins,
  helped-but-source-still-wins, and no-donorward-movement failures. This removes
  the ambiguity in the phrase `controlled backwash`.
- Added a plain numerical definition of exact donor-value error and a class-
  imbalance example explaining balanced accuracy.
- These two new figures are **INCOMPLETE** until notebook 03 is executed on
  Adroit and both outputs are displayed and reviewed in chat. Existing reviewed
  figures and outputs were preserved.
- Interpretation rule: part names are outcomes, not mechanisms. The working
  explanation is the balance between the original-image source advantage and
  part-pixel-driven response. The original margin is not pure context because
  source pixels are still present; later residual tests ask whether species/body
  context helps preserve the source after replacement. Label/visibility conflict,
  exact-value difficulty, alternative frequency, and residual species
  organization. FunnyBird tail is the strongest current example; CUB must rank
  all concepts and mask groups from its own evidence rather than inherit a
  tail-specific claim.

## 2026-08-06: standard-CBM starting-margin correction added

- Notebook 02's accepted controlled-swap result remains valid, but its displayed
  sequence did not give the original margin its own panel. It therefore could
  not distinguish a large starting source advantage from weak donor rise or
  weak release of the removed source.
- Added pending Figure 3b with the exact decomposition
  `m_cf = m_orig + donor_gain + source_decrease` for all five FunnyBird parts.
  `m_orig` is explicitly not called pure context because the original source
  pixels are still present.
- Added pending Figure 4b separating donor wins, donorward-response-but-source-
  still-wins, and no-donorward-response failures for every part.
- Revised accepted interpretations: controlled backwash is a graded five-part
  result. Tail is the most severe FunnyBird observation, not the mechanism;
  beak/eye are substantial and wing/foot retain minority events.
- Figures 3b and 4b are **INCOMPLETE** until executed on Adroit, displayed in
  chat with their tables, and reviewed before any new interpretation is added.
