#!/usr/bin/env bash
# Post-hoc corrected FunnyBird calibration. Run inside an allocated GPU session.
# The original failed run under randomized_patch_masking/ remains untouched.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA first}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
OUT="${PATCH_OUT:-$CURATED_DATA/randomized_patch_masking_v2}"
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
  exit 1
}

FB_OUT="$OUT/funnybirds-cbm-s1.parquet"
CUB_OUT="$OUT/cub70-cbm-s1.parquet"

# Correction 1: relocate each Gaussian patch separately for controls instead of
# translating the complete wide mask rigidly. Count, sigma, and alpha mass match.
# Correction 2: use standardized raw-z response as the primary calibration metric;
# probability is retained as a secondary saturation display.
python analysis/randomized_patch_masking.py \
  --dataset funnybirds --config funnybirds-cbm --seed 1 --epoch "$FB_EPOCH" \
  --funnybirds-root "$CURATED_DATA/FunnyBirds" \
  --max-image-parts-per-part "${FB_MAX_PER_PART:-100}" \
  --control-placement matched_patches \
  --out "$FB_OUT"

python analysis/compare_randomized_patch_masking.py \
  --funnybirds "$FB_OUT" \
  --clean-funnybirds "$CLEAN_FB" \
  --calibration-metric raw_z \
  --out-dir "$OUT/calibration" \
  --fail-on-calibration

echo "[FUNNYBIRD V2 GATE PASSED] starting CUB70 with matched-patch controls"

python analysis/randomized_patch_masking.py \
  --dataset cub70 --config cub70-cbm --seed 1 --epoch "$CUB_EPOCH" \
  --max-image-parts-per-part "${CUB_MAX_PER_PART:-100}" \
  --control-placement matched_patches \
  --out "$CUB_OUT"

python analysis/compare_randomized_patch_masking.py \
  --funnybirds "$FB_OUT" \
  --cub70 "$CUB_OUT" \
  --clean-funnybirds "$CLEAN_FB" \
  --calibration-metric raw_z \
  --out-dir "$OUT/comparison" \
  --fail-on-calibration

echo "[RANDOMIZED PATCH V2 SUITE COMPLETE] $OUT"
