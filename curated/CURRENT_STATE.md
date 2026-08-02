# Current experiment state

Last repository reconciliation: **2026-08-01**, through executed CUB70-CBM
metadata commit `b865608` plus the unexecuted reciprocal-deletion implementation
described below.
Last live Slurm observation: job `3333238` (`fb_rl_broad_s1r`) was running at
1:54:52 on **2026-07-31**. Live cluster state can change and must be refreshed
with `squeue -u "$USER"` before the next cluster decision.

Repository-only update on **2026-08-01**: a reciprocal FunnyBird/CUB70 CBM
deletion suite has been implemented but **not executed**. This update submits no
Slurm job and proves no new biological/model result.

Update this file after any submission, completion, failure, validation,
cancellation, or repository correction. “Submitted” never means “proved.”

## Research order and present decision

| Order | Stage | Present state | Next proof step |
|---:|---|---|---|
| 1 | Non-RL FunnyBirds data | Static dataset claims accepted | No rerun |
| 2 | CBM discovery | Existing FunnyBird results retained; reciprocal shared deletion source added | Execute the shared FunnyBird/CUB70 deletion once, then inspect every intervention and plot |
| 3 | MCBM minimality | Compression/deletion accepted with gamma-saturation caveat; legacy swaps provisional | Finish fixed-cache standard gamma replay and inspect it |
| 4 | RL causal follow-up | Core seed-1 fixed-render notebook executed and 20 figures inspected; broad γ replay running | Finish `3333238`, rebuild all-γ notebook, then fixed-render seeds 2–3 |
| 5 | CUB/CUB70 | Full-resolution CBM notebook executed; all 19 figures inspected; reciprocal deletion source added but unexecuted | Run the calibrated shared deletion, inspect it, then decide the shortest CUB MCBM test |

## Reciprocal FunnyBird/CUB70 CBM test (implemented, unexecuted)

The next shortest proof step is now one command,
`bash analysis/run_paired_deletion.sh`, inside an already allocated GPU session.
It does not train a model and does not submit a cluster job.

| Item | FunnyBird | CUB70 | Acceptance signature/status |
|---|---|---|---|
| Exact positive concept | all five rendered parts | every selected concept with an available mapped mask | implemented; pending execution |
| Target visibility gate | exact part map, at least 0.1% of image | released combined coarse mask, same threshold | implemented; exclusion counts saved |
| Equal-damage control | translated identical mask, nonoverlapping, at least 70% on bird | identical rule | implemented; pairs without a valid control are skipped and counted |
| Four raw-z inputs | original, target-deleted, control-deleted, part-only | identical | implemented; every component retained |
| Calibration | shared deletion versus epoch-100 clean renderer deletion | CUB interpretation depends on this | `funnybird_calibration.status == PASS` required |
| Shared plots | reciprocal section in notebook 02 | identical reciprocal section in notebook 05 | source inserted; execution pending |
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

Acceptance requires both `[PAIRED MASK DELETION PASS]` lines, the calibration
status printed explicitly, `[PAIRED DELETION COMPARISON PASS]`, and final
`[SHARED DELETION SUITE COMPLETE]`. A computational PASS with calibration FAIL
means the CUB inpainting behavior stays quarantined. After execution, export and
display every new figure and intervention sheet in chat before interpretation.

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
| `02_funnybirds_cbm` | Existing discovery chain plus reciprocal FunnyBird/CUB70 deletion section | Existing 18 figures inspected; new 4-figure section unexecuted | CBM discovery remains first; execute shared section without replacing it |
| `03_funnybirds_mcbm` | Full standard-MCBM explanation chain including all-gamma variant confusion | 23 figures inspected; no execution errors | Tail exact-variant attribution does not improve with gamma in seed 1; other parts are mixed; replication pending |
| `03rl_funnybirds_mcbm_relabeled` | Core plus dynamic all-γ and paired-point diagnostics | Core γ=0/0.1 execution inspected; rerun after `3333238` | RL causal follow-up only |
| `04_cub_analysis` | Revised with explicit FunnyBird-data mapping and CUB limits | Executed and exported in `8d65c97`; 3 figures inspected | CUB data stage |
| `05_cub_cbm` | CUB70-CBM exploration plus the identical reciprocal deletion section used in notebook 02; no MCBM or relabeling | Existing 19 figures inspected; new 4-figure section unexecuted. Mask-coverage loss remains fully accounted for. | Execute and visually inspect reciprocal deletion before CUB MCBM |
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
