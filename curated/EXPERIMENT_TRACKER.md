# Experiment tracker

Last reconciled: 2026-07-29. Branch:
`claude/cbm-mcbm-validation-curated-efkd4y`.

Status vocabulary:

- **ACCEPTED** — completed and its evidence currently supports the limited claim.
- **PROVISIONAL** — completed, but a named validation is still missing.
- **QUARANTINED** — computation completed, but its outputs cannot support a claim.
- **PENDING** — required and not yet completed.
- **OPTIONAL** — run only if the corresponding broader claim is retained.

The story order is fixed:

`non-RL data -> CBM discovery -> MCBM minimality -> RL causal follow-up -> CUB`

RL does not replace the non-RL discovery story.

## Stage table

| Order | Stage / question | What has run | Evidence status | What still must run or be checked | Priority |
|---:|---|---|---|---|---:|
| 1 | Non-RL data: what structure exists before modeling? | Notebook 01 dataset analysis; class/concept structure and test species-constancy | **ACCEPTED** for static dataset claims; it never exercised the live renderer | No expensive rerun. Keep its scope explicit: stored data only | Done |
| 2 | CBM discovery: do learned concepts retain/contextualize part information? | Standard CBM seeds 1–3 trained; renderer-free accuracy, deletion, and probe analyses; legacy renderer-swap notebook 02 | Renderer-free results **ACCEPTED**. Legacy swap results **PROVISIONAL** | Regenerate seed-1 swap on a semantically validated fixed cache; inspect every important figure | 2 |
| 3 | MCBM minimality: does the information bottleneck repair the CBM phenomenon? | Standard MCBM gamma 0/0.1 seeds 1–3; gamma 0.3/1/3/5 seed 1; renderer-free compression/deletion analyses; legacy notebook 03 swaps | Renderer-free compression result **ACCEPTED** with gamma-saturation caveat. Legacy swaps **PROVISIONAL** | Use the same validated seed-1 cache as CBM; primary metric is within-image `response_delta`, not absolute post-swap ordering | 3 |
| 4a | RL mechanism: do visibility-aware labels change the causal visual response? | Corrected RLv2 data built; CBM-RLv2 seed 1 and MCBM-RLv2 gamma 0/0.1 seed 1 trained | Training/data **ACCEPTED**; causal swap result **PENDING** | First pass semantic renderer gate; then fixed-cache seed-1 standard/RLv2 comparison at epoch 100 | 1 |
| 4b | RL replication: is the seed-1 direction reproducible? | Jobs 3322224/3322225 submitted and held for seeds 2/3 | **PENDING / HELD** | Leave held until valid seed-1 effect exists. Then finish training and evaluate all seeds on the same validated cache | 4 |
| 4c | RL full-gamma claim | Jobs 3322220–3322223 submitted and held for higher gamma values | **OPTIONAL / HELD** | Run only if claiming an RLv2 effect across the full gamma sweep | Optional |
| 5 | CUB/CUB70: does the mechanism transfer to real birds? | Jobs 3322015/3322016 failed; 3322211 failed; 3322212 completed; one CUB70 CBM parquet export succeeded only under temporary storage | **PARTIAL / INVENTORY REQUIRED** | Inventory checkpoints and durable outputs before rerunning anything; then continue from the shortest missing export/evaluation | After non-RL/RL gate |

## Renderer evidence ledger

| Evaluation | Job / source | Literal observation | Status |
|---|---|---|---|
| Legacy curated CBM examples | Executed notebook 02 HTML | Saved spot-check reported changed pixels: tail 349, wing 827, beak 111, foot 734, eye 177 | Non-degenerate spot-check, but not proof of every legacy render |
| Original MCBM notebook | `fb_mcbm_renderer_swap.ipynb` and `*_wtfres` | Every gamma/part combined to exactly 50%; mean margin zero; forward/backward complementary | Suspicious; cannot support a grounding conclusion |
| Fixed-cache v1 | Job 3329834, epoch 100 | All 15 RGB examples byte-identical and nearly black; all five part maps byte-identical | **QUARANTINED** |
| Fixed-cache v2 | Not run | Must save and display `renderer_preflight/renderer_semantic_preflight.png` before model inference is accepted | **NEXT GATE** |

`$CURATED_DATA/swap_fixed_v1` and its `fixed_rl_comparison.csv` are quarantined.
Do not delete them silently and do not reuse their cache. New work goes under
`$CURATED_DATA/swap_fixed_v2`.

## Slurm job ledger

| Job ID | Job name / purpose | Last known state | Decision |
|---:|---|---|---|
| 3329834 | fixed-cache seed-1 comparison | COMPLETED, exit 0 | **QUARANTINED**: semantic renderer failure |
| 3322224 | CBM-RLv2 seeds 2/3 | PENDING, JobHeldUser | Keep held until valid seed-1 causal response |
| 3322225 | MCBM-RLv2 gamma 0/0.1 seeds 2/3 | PENDING, JobHeldUser | Keep held until valid seed-1 causal response |
| 3322220–3322223 | RLv2 higher gamma | PENDING, JobHeldUser | Optional; keep held |
| 3322015 | CUB CBM attempt | FAILED | Inventory newer checkpoints before rerun |
| 3322016 | CUB70 CBM attempt | FAILED | Inventory newer checkpoints before rerun |
| 3322211 | MCBM sweep attempt | FAILED | Inspect exact completed checkpoints; do not restart wholesale |
| 3322212 | MCBM sweep attempt | COMPLETED | Inspect exact outputs and map to CUB/CUB70 before deciding next |

## Acceptance gate for the next run

The next computation is renderer preflight only. Accept it only if:

1. the log contains `[renderer semantic preflight PASS]`;
2. it exits before model loading and says `[preflight-only]`;
3. stored and live canonical birds are visually consistent;
4. every row in the contact sheet contains a real bird;
5. only the named part changes between original, swap, and deletion;
6. the part map highlights a plausible region for that part.

After acceptance, the seed-1 comparison must additionally:

1. load exactly `epoch_100.pt` for all six comparators;
2. reuse identical valid render IDs and bytes;
3. pass hash diversity, original-versus-counterfactual, and part-map checks;
4. interpret `response_delta =
   (z_donor-z_source)_swap - (z_donor-z_source)_orig` as primary;
5. treat absolute `ordering_correct` as secondary;
6. display and caption every important figure before accepting its interpretation.
