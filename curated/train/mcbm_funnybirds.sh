#!/usr/bin/env bash
# Train MCBM on FunnyBirds via official minimal_cbm bin/train.py.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-42}"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA
MCBM="curated/external/minimal_cbm"
CONFIG="curated/train/configs/funnybirds-mcbm.yaml"
cd "$MCBM"
echo "### MCBM FunnyBirds  config=$CONFIG seed=$SEED"
python3 bin/train.py "../../$CONFIG" -s "$SEED"
echo "Done."
