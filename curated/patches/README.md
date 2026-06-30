# Compatibility patches & shims

We never edit an existing official file inside `external/`. Everything here
is either (a) an import-time shim in `curated/compat/` that our own code
applies, (b) a documented, minimal change recorded below so it can be
reproduced and cited, or (c) new files added at an extension point the
official repo itself designed for user content (e.g. `minimal_cbm/configs/`,
see item 3 below). Each item says **what**, **why**, and **how** to apply it.

If a change is small and local, prefer the shim. Reach for an actual `.patch`
(committed here, applied by `setup.sh`) only when a wrapper genuinely can't
intercept the call.

---

## ConceptBottleneck (CBM)

1. **torchvision Inception weights API.** 2020 code calls
   `models.inception_v3(pretrained=True, ...)`. On torchvision ≥0.13 this warns;
   ≥0.15 it is removed. The pinned `torchvision=0.13.1` in `environment-cbm.yml`
   still accepts `pretrained=`, so **no change needed** at that pin. If you must
   use newer torchvision, replace with `weights=Inception_V3_Weights.IMAGENET1K_V1`.

2. **NumPy aliases.** `np.int`, `np.float`, `np.bool` were removed in NumPy 1.24.
   We pin `numpy=1.23`, so **no change needed**. (Alternative: shim in
   `compat/numpy_compat.py`.)

3. **Pickle protocol / paths in `class_attr_data_10`.** The official
   `data_processing.py` writes absolute image paths into the pickled lists.
   `CUBDataset` already supports path remapping; set the correct image root in
   the training flags (`-data_dir2 CUB`) rather than editing the pickles.

4. **`N_CLASSES` hardcoded to 200 (FunnyBirds support).** `CUB/config.py`
   defines `N_CLASSES = 200` as a module constant; `CUB/train.py` does
   `from CUB.config import ... N_CLASSES ...` at import time and uses it
   directly (`num_classes=N_CLASSES`) when constructing every model. There is
   no CLI flag for it. `patches/run_cbm_funnybirds.py` is a drop-in
   replacement for `experiments.py` that sets `CUB.config.N_CLASSES = 50`
   (and `N_ATTRIBUTES = 26`) *before* `CUB.train` is ever imported, then
   otherwise replicates `experiments.py`'s `__main__` verbatim.
   `train/cbm_funnybirds.sh` calls this wrapper instead of `experiments.py`
   directly; `-n_attributes 26` is passed as an ordinary CLI flag since that
   one IS configurable that way.

## minimal_cbm (MCBM)

1. **Hardcoded `wandb_key` in `bin/train.py`.** `TrainExperiment` is constructed
   with a baked-in wandb key. We do **not** log to that account. Always export
   `WANDB_MODE=offline` (the `train/mcbm_*.sh` wrappers do this) and, if needed,
   `WANDB_DISABLED=true`. No source edit required.

2. **CUB dataset loader scope.** The repo ships `configs/cub12/...` — a 12-concept
   CUB variant on small images. Our CUB/FunnyBirds runs use full attribute sets
   and InceptionV3-scale images, so we author our own configs in
   `train/configs/` against the same YAML schema and (where the bundled encoders
   `conv2d/cifar_resnet/mlp` are too small) set `encoder.arch` to a backbone that
   matches our image size. Verify the chosen arch exists in
   `external/minimal_cbm/src/models/` before training; if not, add it via a
   subclass in `curated/` rather than editing `external/`.

3. **Config lookup cannot take a path.** `BaseExperiment.__init__`
   (`src/experiments/base.py`) hardcodes
   `read_config(os.path.join(root, "configs", config_subpath, config_file))`,
   where `root` is `minimal_cbm`'s *own* repo root and `config_subpath` is
   derived by splitting `config_file` on `"-"` -- it never accepts an
   arbitrary path, and `read_config()` is plain `yaml.load` with no
   env-var/OmegaConf interpolation. `train/configs/cub-mcbm.yaml` and
   `funnybirds-mcbm.yaml` are therefore **templates** (`__CURATED_DATA__`,
   `__GAMMA__` placeholders); `train/mcbm_cub.sh` / `mcbm_funnybirds.sh`
   sed-render them into new files at `external/minimal_cbm/configs/<dataset>/
   <name>.yaml` (a directory the official repo itself ships as an empty,
   user-fillable extension point -- e.g. `configs/cub12/`) before invoking
   `bin/train.py`/`run_mcbm_funnybirds.py` with the bare config name. This is
   the one place curated/ adds files inside `external/`; no existing official
   file is ever modified, and the rendered file is regenerated per job.

4. **FunnyBirds has zero upstream dataset support.** `src/datasets/__init__.py`'s
   `get_loader()` is a hardcoded if/elif with branches only for
   `CUB200, CIFAR10, DSPRITES/MPI3D/SHAPES3D, CELEBA, SPIRALS`,  and
   `src/experiments/train.py` does `from src.datasets import get_loader` at
   import time. `compat/funnybirds_mcbm_dataset.py` implements a
   `FunnyBirdsMCBM(Dataset)` + `get_funnybirds()` mirroring `cub200.py`'s
   interface exactly (including the `return_nuisances=True` 5-tuple shape
   `TrainExperiment._set_loaders`/`train_epoch` unconditionally expect --
   FunnyBirds has no held-out nuisance attributes, so those two extra tensors
   are always empty). `patches/run_mcbm_funnybirds.py` monkey-patches
   `src.datasets.get_loader` to add a `"FUNNYBIRDS"` branch *before*
   `src.experiments` is ever imported, then replicates `bin/train.py`'s
   `__main__` verbatim. `train/mcbm_funnybirds.sh` calls this wrapper instead
   of `bin/train.py` directly.
