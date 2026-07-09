#!/usr/bin/env bash
# BASE CASE FIRST: train vanilla (no concepts) and CBM (reference) BEFORE the
# MCBM gamma sweep, through the SAME official trainer + SAME backbone, so every
# analysis instrument is validated on the reference before the treatment and the
# CBM-vs-MCBM comparison has no backbone/preprocessing confound.
#
# Usage:
#   bash curated/train/run_baselines.sh funnybirds            # vanilla + cbm, seed 1
#   MODELS="cbm" SEEDS="1 2 3" bash curated/train/run_baselines.sh funnybirds
#   ARCH=inception_v3 MODELS="cbm" bash curated/train/run_baselines.sh cub   # crossed backbone
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"

DATASET="${1:?usage: run_baselines.sh <funnybirds|cub|cub70>}"
MODELS="${MODELS:-vanilla cbm}"            # base case = vanilla + cbm; drop vanilla if not needed
SEEDS="${SEEDS:-1}"
ARCH="${ARCH:-resnet50}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"
source "$HERE/_paths.sh"
GEN_DIR="$MCBM/configs/${DATASET}"; mkdir -p "$GEN_DIR"

echo "### baselines  dataset=$DATASET  arch=$ARCH  models=[$MODELS]  seeds=[$SEEDS]"
for m in $MODELS; do
  tmpl="$HERE/configs/${DATASET}-${m}.yaml"
  base="${DATASET}-${m}"
  cfg="$GEN_DIR/${base}.yaml"
  gen_config "$tmpl" "$cfg" "$DATASET" "$ARCH" "" || exit 1
  for s in $SEEDS; do
    echo ">>> model=$m seed=$s  ($base)"
    ( cd "$HERE" && python3 run_mcbm.py "$base" -s "$s" )
  done
done
echo "Done. Results in $MCBM/results/<name>/<seed>/.  Analyze CBM fully before the MCBM sweep."
