#!/usr/bin/env bash
# Single MCBM/FunnyBirds run = a 1-point gamma sweep, so it goes through the same
# verified path (run_mcbm.py: FunnyBirds loader shim + wandb-offline shim). The
# old version called bin/train.py directly, which has NO FunnyBirds loader and
# inits wandb ONLINE with a hardcoded key -- do not resurrect that.
#   usage: bash mcbm_funnybirds.sh [SEED=42] [GAMMA=1]
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-42}"
GAMMA="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAMMAS="$GAMMA" SEEDS="$SEED" bash "$HERE/mcbm_gamma_sweep.sh" funnybirds
