#!/usr/bin/env bash
# Same raw-z intervention on both datasets. Run inside an allocated GPU session.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA first}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
OUT="${PAIRED_OUT:-$CURATED_DATA/paired_deletion}"
FB_EPOCH="${FB_EPOCH:-100}"
# The current accepted cub70-cbm export was made from epoch 250. Override only
# when CURRENT_STATE records a different selected checkpoint.
CUB_EPOCH="${CUB_EPOCH:-250}"
mkdir -p "$OUT"
cd "$CURATED"

FB_CKPT="external/minimal_cbm/results/funnybirds-cbm/1/models/epoch_${FB_EPOCH}.pt"
CUB_CKPT="external/minimal_cbm/results/cub70-cbm/1/models/epoch_${CUB_EPOCH}.pt"
for checkpoint in "$FB_CKPT" "$CUB_CKPT"; do
  test -s "$checkpoint" || { echo "MISSING CHECKPOINT: $checkpoint"; exit 1; }
done

# Exact epoch-matched clean FunnyBird reference. This uses the original renderer
# interventions and is the calibration target for the shared mask inpainting.
CLEAN_FB="$OUT/funnybirds-clean-renderer-epoch${FB_EPOCH}-s1.parquet"
if [ ! -s "$CLEAN_FB" ]; then
  python analysis/grounding_deletion.py \
    --config funnybirds-cbm --seed 1 --epoch "$FB_EPOCH" \
    --funnybirds-root "$CURATED_DATA/FunnyBirds" \
    --pkls "$CURATED_DATA/funnybirds_processed" \
    --out "$CLEAN_FB"
fi

python analysis/paired_mask_deletion.py \
  --dataset funnybirds --config funnybirds-cbm --seed 1 --epoch "$FB_EPOCH" \
  --out "$OUT/funnybirds-cbm-s1.parquet"

python analysis/paired_mask_deletion.py \
  --dataset cub70 --config cub70-cbm --seed 1 --epoch "$CUB_EPOCH" \
  --out "$OUT/cub70-cbm-s1.parquet"

python analysis/compare_paired_mask_deletion.py \
  --funnybirds "$OUT/funnybirds-cbm-s1.parquet" \
  --cub70 "$OUT/cub70-cbm-s1.parquet" \
  --clean-funnybirds "$CLEAN_FB" \
  --out-dir "$OUT/comparison"

echo "[SHARED DELETION SUITE COMPLETE] $OUT"
