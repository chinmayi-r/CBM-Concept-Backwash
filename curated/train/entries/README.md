# Explicit seed-1 entries

There is no logic-bearing bulk campaign launcher. Run one script for one
completion-matrix entry. Each submission script prints its complete payload and
the returned Slurm job id.

`submit_all_current_seed1.sh` is now a fail-closed tombstone: it submits
nothing and points to entries 05–07. This prevents the former campaign command
from launching superseded epoch-100 work.

| Order | Script | Work performed |
|---:|---|---|
| 0 | `00_reconcile_funnybird_standard_s1.sh` | Validate the already-completed 100-epoch artifact and create its transparent acceptance manifest. No GPU job. |
| 1 | `01_submit_funnybird_rlv2_s1.sh` | Train Koh Joint ResNet-50 `accelerated_v1` on the frozen RLv2 label view; validate; export milestone/final test tables; write `SUCCESS.json`. |
| 2 | `02_submit_cub70_standard_s1.sh` | Train the 70-class/112-concept Koh Joint ResNet-50 model; validate; export test table; write `SUCCESS.json`. |
| 3 | `03_submit_full_cub_standard_s1.sh` | Train the 200-class/112-concept Koh Joint ResNet-50 model; validate; export test table; write `SUCCESS.json`. |
| 4 | `04_submit_funnybird_swaps_s1.sh [RL_JOB_ID]` | After both FunnyBird models, run fixed renders, validate swaps, compare standard/RLv2, and write the swap manifest. |
| 5 | `05_submit_funnybird_standard_convergence_s1.sh` | Resume Standard from epoch 100 in 25-epoch blocks at the terminal LR, stopping on the unchanged stability gate or at epoch 200. |
| 6 | `06_submit_funnybird_rlv2_convergence_s1.sh` | Resume RLv2 identically from epoch 100, stopping on the unchanged stability gate or at epoch 200. |
| 7 | `07_submit_funnybird_converged_swaps_s1.sh STANDARD_JOB_ID RLV2_JOB_ID` | Run the fixed swaps only after both matched convergence continuations are accepted. |

Entries 0–4 document the preceding epoch-100 campaign and must not be rerun.
Entries 5 and 6 are the two current independent jobs. Entry 7 is their only
consumer and uses `afterok` for whichever continuation manifests are not yet
present.

The training jobs themselves perform the visible sequence
`audit -> train -> checkpoint validation -> test extraction -> manifest` in
`koh_joint_stage.sh`. FunnyBird additionally performs the four milestone
exports and convergence audit. A nonzero exit in one independent job does not
cancel either of the other independent jobs.

This directory intentionally contains no seed-2/3 launchers, no CUB70 MCBM
gamma-3/5 retry, and no full-CUB MCBM sweep.

Run only the current entries individually from the repository root:

```bash
bash curated/train/entries/05_submit_funnybird_standard_convergence_s1.sh
bash curated/train/entries/06_submit_funnybird_rlv2_convergence_s1.sh
bash curated/train/entries/07_submit_funnybird_converged_swaps_s1.sh STANDARD_JOB_ID RLV2_JOB_ID
```
