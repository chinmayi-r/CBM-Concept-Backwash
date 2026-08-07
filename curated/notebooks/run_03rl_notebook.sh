#!/usr/bin/env bash
# Rebuild, execute, and export the matched fixed-render RLv2 notebook.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

cat >&2 <<'EOF'
INCOMPLETE: notebook 03rl is intentionally disabled until explicitly seeded
standard/RLv2 MCBM checkpoints and their fixed-render evaluations replace the
legacy unseeded runs. The validated RGB render cache itself remains reusable.
EOF
exit 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

export FIXED_SWAP_DIR="${FIXED_SWAP_DIR:-$CURATED_DATA/swap_fixed_v3_matched}"
RL_TAG="${RL_TAG:-rlv2matched}"

required=(
  "$FIXED_SWAP_DIR/funnybirds-cbm-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-cbm-${RL_TAG}-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g0-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g0-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g0p1-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g0p1-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g0p3-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g0p3-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g1-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g1-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g3-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g3-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-g5-s1.csv"
  "$FIXED_SWAP_DIR/funnybirds-mcbm-${RL_TAG}-g5-s1.csv"
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
python analysis/add_predicate_proof_ladders.py --only 03rl

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/03rl_funnybirds_mcbm_relabeled.ipynb

jupyter nbconvert --to html \
  notebooks/03rl_funnybirds_mcbm_relabeled.ipynb

echo "Executed and exported notebooks/03rl_funnybirds_mcbm_relabeled.html"
