#!/usr/bin/env bash
# Rebuild, execute, and export the matched fixed-render RLv2 notebook.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

export FIXED_SWAP_DIR="${FIXED_SWAP_DIR:-$CURATED_DATA/swap_fixed_v2_attempt2}"
RL_TAG="${RL_TAG:-rlv2matched}"

required=(
  "$FIXED_SWAP_DIR/funnybirds-cbm-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-cbm-${RL_TAG}-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g0-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g0-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g0p1-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g0p1-s1.csv"
  "$FIXED_SWAP_DIR/renderer_preflight/renderer_semantic_preflight.png"
)

missing=0
for path in "${required[@]}"; do
  if [ ! -s "$path" ]; then
    echo "MISSING: $path" >&2
    missing=1
  fi
done
if [ "$missing" = 1 ]; then
  echo "Stop: the validated fixed-render seed-1 comparison is incomplete." >&2
  exit 1
fi

python analysis/build_03rl_notebook.py

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/03rl_funnybirds_mcbm_relabeled.ipynb

jupyter nbconvert --to html \
  notebooks/03rl_funnybirds_mcbm_relabeled.ipynb

echo "Executed and exported notebooks/03rl_funnybirds_mcbm_relabeled.html"
