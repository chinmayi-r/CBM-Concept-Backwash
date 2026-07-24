#!/usr/bin/env bash
# Prepare mask visibility, evaluation-label diagnostics, and eval tables after
# the full-CUB/CUB70 CBM or MCBM checkpoints exist.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"

python3 "$CURATED/data/cub70/build_cub70_visibility.py" --data-root "$CURATED_DATA"
RELABEL_ARGS=(--data-root "$CURATED_DATA"
  --data-dir "CUB_processed/class_attr_data_10_cub70_original")
if [ -n "${CUB_ROOT:-}" ]; then RELABEL_ARGS+=(--cub-root "$CUB_ROOT"); fi
if [ -n "${CUB_ATTR_FILE:-}" ]; then RELABEL_ARGS+=(--attr-names "$CUB_ATTR_FILE"); fi
python3 "$CURATED/data/cub70/relabel_cub_with_cub70.py" "${RELABEL_ARGS[@]}"

CONFIGS="${CONFIGS:-cub-cbm cub70-cbm}"
SEEDS="${SEEDS:-1}"
OUT="$CURATED_DATA/cub70_eval"; mkdir -p "$OUT"
for config in $CONFIGS; do
  for seed in $SEEDS; do
    python3 "$HERE/cub70_export_eval.py" --config "$config" --seed "$seed" \
      --out "$OUT/${config}-s${seed}.parquet"
  done
done
echo "Done -> $OUT"
