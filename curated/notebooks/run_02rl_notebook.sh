#!/usr/bin/env bash
# Build, execute, and export the CBM-only matched RLv2 causal notebook.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

export FIXED_SWAP_DIR="${FIXED_SWAP_DIR:-$CURATED_DATA/swap_fixed_v3_matched}"

required=(
  "$FIXED_SWAP_DIR/funnybirds-cbm-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-cbm-rlv2matched-s1.csv"
  "$FIXED_SWAP_DIR/renderer_preflight/renderer_semantic_preflight.png"
  "external/minimal_cbm/configs/funnybirds/funnybirds-cbm.yaml"
  "external/minimal_cbm/configs/funnybirds/funnybirds-cbm-rlv2matched.yaml"
  "external/minimal_cbm/results/funnybirds-cbm/1/models/epoch_100.pt"
  "external/minimal_cbm/results/funnybirds-cbm-rlv2matched/1/models/epoch_100.pt"
  "external/minimal_cbm/results/funnybirds-cbm/1/predictions/epoch_100.pth"
  "external/minimal_cbm/results/funnybirds-cbm-rlv2matched/1/predictions/epoch_100.pth"
)

missing=0
for path in "${required[@]}"; do
  if [ ! -s "$path" ]; then
    echo "INCOMPLETE: missing $path" >&2
    missing=1
  fi
done
if [ "$missing" = 1 ]; then
  echo "Stop: the matched CBM-only RLv2 report inputs are incomplete." >&2
  exit 1
fi

python analysis/build_02rl_notebook.py

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/02rl_funnybirds_cbm_relabeled.ipynb

jupyter nbconvert --to html \
  notebooks/02rl_funnybirds_cbm_relabeled.ipynb

python analysis/repair_nbconvert_alt_text.py \
  notebooks/02rl_funnybirds_cbm_relabeled.ipynb \
  notebooks/02rl_funnybirds_cbm_relabeled.html

echo "Executed and exported notebooks/02rl_funnybirds_cbm_relabeled.html"
