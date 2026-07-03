#!/usr/bin/env bash
# Comprehensive MCBM gamma sweep on the OFFICIAL minimal_cbm trainer.
#
# This is the scientific spine of the curated restart: the same official code,
# trained from scratch across a wide gamma grid x seeds, so every downstream
# result (recall gap, causal swap, grounding) can be plotted AGAINST gamma. The
# old hand-written MCBM used a single, mis-scaled gamma; this fixes both.
#
# gamma is the minimality/bottleneck strength. Effective z-force = gamma * 0.2
# (the 0.2 is fixed inside minimal_cbm's z_loss). gamma=0 = no minimality (control).
#
# Usage:
#   bash curated/train/mcbm_gamma_sweep.sh funnybirds
#   GAMMAS="0 0.1 0.3 1 3 10 30" SEEDS="1 2 3" bash curated/train/mcbm_gamma_sweep.sh cub
#
# It writes one generated config per gamma to train/configs/_sweep/ and runs the
# official trainer for each (gamma, seed). Output/checkpoint location is whatever
# minimal_cbm writes by default.
#
# >>> CONFIRM ON ADROIT (once): open external/minimal_cbm/bin/train.py and check
#     where it saves checkpoints/logs. If it keys the run dir only by seed (not by
#     config name/gamma), sweeps will COLLIDE. If so, set the run-name/output key
#     in the template config and add it to the sed block below so each gamma gets
#     its own dir. Do this before launching the full grid.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"

DATASET="${1:?usage: mcbm_gamma_sweep.sh <funnybirds|cub|cub70>}"
GAMMAS="${GAMMAS:-0 0.1 0.3 1 3 10 30}"   # wide, log-spaced; includes 0 control
SEEDS="${SEEDS:-1 2 3}"                    # >=3 for error bars on 'X vs gamma'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # curated/train
REPO="$(cd "$HERE/../.." && pwd)"
TEMPLATE="$HERE/configs/${DATASET}-mcbm.yaml"
[ -f "$TEMPLATE" ] || { echo "no template: $TEMPLATE" >&2; exit 1; }
GEN_DIR="$HERE/configs/_sweep"; mkdir -p "$GEN_DIR"
MCBM="$REPO/curated/external/minimal_cbm"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA

echo "### MCBM gamma sweep  dataset=$DATASET  gammas=[$GAMMAS]  seeds=[$SEEDS]"
for g in $GAMMAS; do
  gtag="${g//./p}"                                          # 0.1 -> 0p1 (dir-safe)
  cfg="$GEN_DIR/${DATASET}-mcbm-g${gtag}.yaml"
  # swap ONLY the gamma line; everything else is the verified template verbatim
  sed -E "s/^([[:space:]]*)gamma:[[:space:]]*[0-9.]+.*/\1gamma: ${g}/" "$TEMPLATE" > "$cfg"
  grep -qE "gamma:[[:space:]]*${g}([^0-9]|$)" "$cfg" || { echo "gamma sub failed for $g" >&2; exit 1; }
  for s in $SEEDS; do
    echo ">>> gamma=$g seed=$s  ($cfg)"
    ( cd "$MCBM" && python3 bin/train.py "$cfg" -s "$s" )
  done
done
echo "Done. Sweep configs in $GEN_DIR ; dump eval tables per run next (see analysis/io.py)."
