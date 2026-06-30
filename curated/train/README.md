# Training wrappers

These are **thin** wrappers over the official entry points — no training loop is
reimplemented. Set `CURATED_DATA` and run from the repo root inside the right
conda env. Each script echoes the underlying official command before running it,
so the exact provenance is in the logs.

| Script | Model | Dataset | Underlying entry point |
|--------|-------|---------|------------------------|
| `cbm_cub.sh`        | CBM  | CUB-200 | `external/ConceptBottleneck/src/experiments.py cub <Mode>` |
| `cbm_cub70.sh`      | CBM  | CUB70 (70-class) | same, on class-filtered pickles |
| `cbm_funnybirds.sh` | CBM  | FunnyBirds | same, on `funnybirds_processed` pickles |
| `mcbm_cub.sh`        | MCBM | CUB-200 | `external/minimal_cbm/bin/train.py <config> -s <seed>` |
| `mcbm_cub70.sh`      | MCBM | CUB70 | same, cub70 config |
| `mcbm_funnybirds.sh` | MCBM | FunnyBirds | same, funnybirds config |

Flags in `cbm_*.sh` are copied verbatim from
`external/ConceptBottleneck/CUB/README.md` (the three regimes: Independent,
Sequential, Joint). **Before the first real run, diff them against that README**
in case the pinned commit differs. Choose ONE regime to report and say which in
the paper; the scripts run all three so you can compare leakage.

MCBM configs live in `configs/`. `mcbm_*.sh` exports `WANDB_MODE=offline` to
neutralize the hardcoded key (patches/README #2).
