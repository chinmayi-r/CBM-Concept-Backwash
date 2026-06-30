# Compatibility patches & shims

We never edit files inside `external/`. Everything here is either (a) an
import-time shim in `curated/compat/` that our own code applies, or (b) a
documented, minimal change recorded below so it can be reproduced and cited.
Each item says **what**, **why**, and **how** to apply it.

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
