#!/usr/bin/env bash
set -euo pipefail

: "${CURATED_DATA:?Set CURATED_DATA to the curated data directory}"

OUT_ROOT="${PILOT_OUT:-$CURATED_DATA/cub70_beak_tail_swap_pilot}"
mkdir -p "$OUT_ROOT"

python analysis/cub70_beak_tail_swap_pilot.py \
  --config cub70-cbm \
  --seed 1 \
  --epoch 250 \
  --out "$OUT_ROOT/cub70-cbm-s1.parquet"

echo "[CUB70 BEAK/TAIL PILOT DRIVER COMPLETE] $OUT_ROOT"
