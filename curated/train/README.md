# Current training interface

Use only the one-entry scripts in [`entries/`](entries/README.md) for the
current seed-1 standard-CBM work. There is no bulk launcher.

Each entry script prints:

- dataset, labels, seed, species count, and concept count;
- framework and backbone;
- loss and training protocol;
- output root and dependency (`none` for every independent training entry);
- the ordered work performed by the compute job;
- the submitted Slurm job id and the payload Slurm accepted.

The compute jobs use these shared implementations:

| File | Responsibility |
|---|---|
| `koh_accelerated_funnybird_seed1_job.slurm` | GPU resources and interruption handling for FunnyBird `accelerated_v1`. |
| `koh_joint_job.slurm` | GPU resources and interruption handling for CUB70/full-CUB Koh training. |
| `koh_joint_stage.sh` | Audits, invokes Koh Joint training, validates the checkpoint, extracts test outputs, and writes the manifest. |
| `koh_funnybird_seed1_swaps_job.slurm` | GPU resources for the accepted two-model fixed-swap evaluation. |
| `koh_funnybird_seed1_swaps_stage.sh` | Verifies both model manifests, evaluates swaps, validates outputs, compares standard/RLv2, and writes the swap manifest. |

The many older `cbm_*.sh`, `mcbm_*.sh`, sweep, baseline, provisional, and bulk
submission files are historical provenance. They are not the current interface
and must not be substituted for the explicit entry scripts. In particular,
`minimal_cbm` CBM launchers are not valid standard-CBM training.
