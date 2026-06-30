#!/usr/bin/env bash
# Train MCBM on CUB-200 via the official minimal_cbm bin/train.py.
# WANDB_MODE=offline neutralizes the hardcoded wandb key (patches/README #2).
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-42}"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA
MCBM="curated/external/minimal_cbm"
CONFIG="curated/train/configs/cub-mcbm.yaml"
cd "$MCBM"
echo "### MCBM CUB-200  config=$CONFIG seed=$SEED"
python3 bin/train.py "../../$CONFIG" -s "$SEED"
echo "Done."
