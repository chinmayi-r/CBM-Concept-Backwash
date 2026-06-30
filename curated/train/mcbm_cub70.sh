#!/usr/bin/env bash
# Train MCBM on the 70-class CUB70 subset (prof note #3/#4).
# LABELS=original|relabeled selects the manifest (build via build_cub_mcbm_data.py
# with --data-dir class_attr_data_10[_relabeled], then filter to classes 0..69).
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-42}"
LABELS="${2:-original}"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA LABELS
MCBM="curated/external/minimal_cbm"
CONFIG="curated/train/configs/cub70-mcbm.yaml"   # reads $LABELS to pick the manifest
cd "$MCBM"
echo "### MCBM CUB70 labels=$LABELS config=$CONFIG seed=$SEED"
python3 bin/train.py "../../$CONFIG" -s "$SEED"
echo "Done."
