# Training wrappers

These are **SLURM batch scripts** (Adroit conventions: `-p gpu`,
`--gres=gpu:1`, `module load anaconda3/2025.6`) over the official entry
points -- no training loop is reimplemented. Submit with `sbatch`, with
`CURATED_DATA` set in your environment first. Separate conda envs for CBM
(`cbm`) vs MCBM (`mcbm`) -- see `curated/README.md` -- *not* the real
pipeline's single `cubvision-gpu` env. Each script echoes the underlying
official command before running it, so the exact provenance is in the logs.

| Script | Model | Dataset | Underlying entry point |
|--------|-------|---------|------------------------|
| `cbm_cub.sh`         | CBM  | CUB-200 | `external/ConceptBottleneck/experiments.py cub <Mode>` (repo-root `experiments.py`, not `src/experiments.py`) |
| `cbm_cub70.sh`       | CBM  | CUB70 (70-class) | same, on class-filtered pickles |
| `cbm_funnybirds.sh`  | CBM  | FunnyBirds | `patches/run_cbm_funnybirds.py cub <Mode>` -- patches `CUB.config.N_CLASSES` 200→50 before `CUB.train` is imported (official repo hardcodes 200; see that file's docstring) |
| `mcbm_cub.sh`        | MCBM | CUB-200 | `external/minimal_cbm/bin/train.py cub-mcbm -s <seed>`, SLURM array `0-7` sweeping `gamma` |
| `mcbm_cub70.sh`      | MCBM | CUB70 | same, cub70 config |
| `mcbm_funnybirds.sh` | MCBM | FunnyBirds | `patches/run_mcbm_funnybirds.py funnybirds-mcbm -s <seed>` -- registers a FunnyBirds dataset into `src.datasets.get_loader` (official repo has zero FunnyBirds support; see `compat/funnybirds_mcbm_dataset.py`), SLURM array `0-7` sweeping `gamma` |

Flags in `cbm_*.sh` are copied verbatim from
`external/ConceptBottleneck/CUB/README.md` (the four regimes: Concept_XtoC,
Independent, Sequential, Joint). **Before the first real run, diff them
against that README** in case the pinned commit differs.

MCBM gamma sweep (`mcbm_cub.sh`, `mcbm_funnybirds.sh`):
`GAMMAS=(0.0 0.05 0.1 0.2 0.5 1.0 2.5 5.0)`, mirroring (and extending) the
real pipeline's `run_mcbm_train_adroit.sh` / `run_fb_mcbm_train_adroit.sh`
arrays.

MCBM configs live in `configs/*-mcbm.yaml` as **templates**: `minimal_cbm`'s
`BaseExperiment.__init__` hardcodes config lookup to
`<minimal_cbm_root>/configs/<name.split('-')[0]>/<name>.yaml` and
`read_config()` is plain `yaml.load` (no env-var interpolation), so
`mcbm_*.sh` sed-renders the `__CURATED_DATA__`/`__GAMMA__` placeholders into a
real file inside `external/minimal_cbm/configs/` before each run -- see
`curated/README.md`'s "narrow, documented exception" note. `mcbm_*.sh` also
exports `WANDB_MODE=offline` to neutralize the hardcoded key (patches/README
#2).
