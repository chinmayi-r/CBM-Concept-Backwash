#!/usr/bin/env bash
# Run inside an already allocated GPU session. No sbatch is issued here.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA first}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
OUT="${PATCH_OUT:-$CURATED_DATA/randomized_patch_masking}"
FB_EPOCH="${FB_EPOCH:-100}"
CUB_EPOCH="${CUB_EPOCH:-250}"
mkdir -p "$OUT/calibration" "$OUT/comparison"
cd "$CURATED"

FB_CKPT="external/minimal_cbm/results/funnybirds-cbm/1/models/epoch_${FB_EPOCH}.pt"
CUB_CKPT="external/minimal_cbm/results/cub70-cbm/1/models/epoch_${CUB_EPOCH}.pt"
for checkpoint in "$FB_CKPT" "$CUB_CKPT"; do
  test -s "$checkpoint" || { echo "MISSING CHECKPOINT: $checkpoint"; exit 1; }
done

CLEAN_FB="$CURATED_DATA/paired_deletion/funnybirds-clean-renderer-epoch${FB_EPOCH}-s1.parquet"
test -s "$CLEAN_FB" || {
  echo "MISSING ACCEPTED CLEAN FUNNYBIRD REFERENCE: $CLEAN_FB"
  echo "Do not regenerate or substitute it silently."
  exit 1
}

FB_OUT="$OUT/funnybirds-cbm-s1.parquet"
CUB_OUT="$OUT/cub70-cbm-s1.parquet"

# Stage 1: FunnyBird only. The comparison exits 2 if any preregistered
# calibration check fails. set -e then prevents the CUB command from running.
python analysis/randomized_patch_masking.py \
  --dataset funnybirds --config funnybirds-cbm --seed 1 --epoch "$FB_EPOCH" \
  --funnybirds-root "$CURATED_DATA/FunnyBirds" \
  --max-image-parts-per-part "${FB_MAX_PER_PART:-100}" \
  --out "$FB_OUT"

python analysis/compare_randomized_patch_masking.py \
  --funnybirds "$FB_OUT" \
  --clean-funnybirds "$CLEAN_FB" \
  --out-dir "$OUT/calibration" \
  --fail-on-calibration

echo "[FUNNYBIRD GATE PASSED] starting CUB70 with the identical mask settings"

# Stage 2 runs only after Stage 1 passes.
python analysis/randomized_patch_masking.py \
  --dataset cub70 --config cub70-cbm --seed 1 --epoch "$CUB_EPOCH" \
  --max-image-parts-per-part "${CUB_MAX_PER_PART:-100}" \
  --out "$CUB_OUT"

python analysis/compare_randomized_patch_masking.py \
  --funnybirds "$FB_OUT" \
  --cub70 "$CUB_OUT" \
  --clean-funnybirds "$CLEAN_FB" \
  --out-dir "$OUT/comparison" \
  --fail-on-calibration

echo "[RANDOMIZED PATCH SUITE COMPLETE] $OUT"
