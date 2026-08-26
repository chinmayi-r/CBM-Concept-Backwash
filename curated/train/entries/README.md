# Explicit seed-1 entries

There is no bulk campaign launcher. Run one script for one completion-matrix
entry. Each submission script prints its complete payload and the returned
Slurm job id.

| Order | Script | Work performed |
|---:|---|---|
| 0 | `00_reconcile_funnybird_standard_s1.sh` | Validate the already-completed 100-epoch artifact and create its transparent acceptance manifest. No GPU job. |
| 1 | `01_submit_funnybird_rlv2_s1.sh` | Train Koh Joint ResNet-50 `accelerated_v1` on the frozen RLv2 label view; validate; export milestone/final test tables; write `SUCCESS.json`. |
| 2 | `02_submit_cub70_standard_s1.sh` | Train the 70-class/112-concept Koh Joint ResNet-50 model; validate; export test table; write `SUCCESS.json`. |
| 3 | `03_submit_full_cub_standard_s1.sh` | Train the 200-class/112-concept Koh Joint ResNet-50 model; validate; export test table; write `SUCCESS.json`. |
| 4 | `04_submit_funnybird_swaps_s1.sh [RL_JOB_ID]` | After both FunnyBird models, run fixed renders, validate swaps, compare standard/RLv2, and write the swap manifest. |

Entries 1–3 are independent Slurm jobs. They have no dependencies on one
another. Entry 4 is the only consumer: if RLv2 is still running, pass its job
id and the script uses `afterok` for that job; otherwise it requires the
completed RLv2 manifest.

The training jobs themselves perform the visible sequence
`audit -> train -> checkpoint validation -> test extraction -> manifest` in
`koh_joint_stage.sh`. FunnyBird additionally performs the four milestone
exports and convergence audit. A nonzero exit in one independent job does not
cancel either of the other independent jobs.

This directory intentionally contains no seed-2/3 launchers, no CUB70 MCBM
gamma-3/5 retry, and no full-CUB MCBM sweep.

Run entries individually from the repository root:

```bash
bash curated/train/entries/00_reconcile_funnybird_standard_s1.sh
bash curated/train/entries/01_submit_funnybird_rlv2_s1.sh
bash curated/train/entries/02_submit_cub70_standard_s1.sh
bash curated/train/entries/03_submit_full_cub_standard_s1.sh
# Use the job id printed by entry 1 while RLv2 is still running:
bash curated/train/entries/04_submit_funnybird_swaps_s1.sh RLV2_JOB_ID
```
