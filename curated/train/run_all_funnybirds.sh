#!/usr/bin/env bash
# FIRE-AND-FORGET: the whole FunnyBirds spine end-to-end, idempotent/resumable.
#   1. baselines (vanilla + cbm)          -- skipped if already trained
#   2. MCBM gamma sweep (the foil)        -- skipped per-gamma if already trained
#   3. deletion grounding on every model  -- skipped if already scored
#   4. collect BACKWASH vs gamma -> $CURATED_DATA/backwash_vs_gamma.{csv,png,pdf}
#
# Re-running resumes: a run counts as "trained" once its FINAL checkpoint exists
# (epoch_${N_EPOCHS}.pt). If a job hits the time wall mid-gamma, just resubmit.
#
#   GAMMAS="0 1 3 10 30" SEEDS="1" bash train/run_all_funnybirds.sh
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
export ARCH="${ARCH:-resnet50}"
GAMMAS="${GAMMAS:-0 0.3 1 3 10 30}"
SEEDS="${SEEDS:-1}"
N_EPOCHS="${N_EPOCHS:-100}"          # must match train/configs/funnybirds-*.yaml
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # curated/train
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"

trained() { [ -f "$MCBM/results/$1/$2/models/epoch_${N_EPOCHS}.pt" ]; }

echo "### [1/4] baselines (vanilla, cbm) ###"
for m in vanilla cbm; do for s in $SEEDS; do
  if trained "funnybirds-$m" "$s"; then echo "  skip funnybirds-$m s$s (trained)"; continue; fi
  MODELS="$m" SEEDS="$s" bash "$HERE/run_baselines.sh" funnybirds
done; done

echo "### [2/4] MCBM gamma sweep ###"
for g in $GAMMAS; do gtag="${g//./p}"; base="funnybirds-mcbm-g${gtag}"
  for s in $SEEDS; do
    if trained "$base" "$s"; then echo "  skip $base s$s (trained)"; continue; fi
    GAMMAS="$g" SEEDS="$s" bash "$HERE/mcbm_gamma_sweep.sh" funnybirds
  done
done

echo "### [3+4/4] grounding on all + collect ###"
bash "$CURATED/analysis/grounding_sweep.sh"

echo "ALL DONE -> $CURATED_DATA/backwash_vs_gamma.{csv,png,pdf}"
