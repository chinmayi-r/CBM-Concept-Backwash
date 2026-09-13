#!/usr/bin/env bash
# Frozen CUB70-mask/Grad-CAM audit for accepted official Koh Joint models.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"

CURATED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CURATED"

mode="${1:-cub70}"
case "$mode" in
  cub70)
    DATASET=cub70
    MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_v1/cub70/standard/seed1"
    DATA_PKL="$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/test.pkl"
    WORK_DIR="$CURATED_DATA/koh_joint_inputs/work/cub70"
    OUT_DIR="$CURATED_DATA/cub_koh_spatial_v1/cub70_standard_s1"
    ;;
  full)
    DATASET=cub
    MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_decay_continuation_v1/cub/standard/seed1"
    DATA_PKL="$CURATED_DATA/CUB_processed/class_attr_data_10/test.pkl"
    WORK_DIR="$CURATED_DATA/koh_joint_inputs/work/cub"
    OUT_DIR="$CURATED_DATA/cub_koh_spatial_v1/full_cub_standard_s1"
    ;;
  *) echo "ERROR: mode must be cub70 or full" >&2; exit 2 ;;
esac

echo "GOAL: test whether each accepted Koh concept logit is spatially concentrated on its named CUB70 mask"
echo "MODEL: $mode Standard Koh Joint ResNet-50 seed 1"
echo "METHOD: concept-specific Grad-CAM plus released-mask overlap; frozen saved-head magnitude replacement"
echo "TRAINING: no"
echo "BOUNDARY: post-hoc localization diagnostic, not an exact segment contribution and not a donor-part swap"
echo "OUTPUT: $OUT_DIR"

python analysis/test_cub_koh_spatial_audit.py
python analysis/cub_koh_spatial_audit.py \
  --model-root "$MODEL_ROOT" \
  --data-pkl "$DATA_PKL" \
  --work-dir "$WORK_DIR" \
  --mask-root "$CURATED_DATA/cub70/masks" \
  --visibility "$CURATED_DATA/cub70_visibility.parquet" \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR"
