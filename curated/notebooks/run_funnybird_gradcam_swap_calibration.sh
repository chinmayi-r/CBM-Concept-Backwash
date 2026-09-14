#!/usr/bin/env bash
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA to the shared curated_data directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

MODEL="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/best_model_1.pth"
SWAPS="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1/funnybirds-cbm-s1.csv"
OUT="$CURATED_DATA/funnybird_gradcam_swap_calibration_v1/standard_seed1"
FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-$CURATED_DATA/FunnyBirds}"

echo "GOAL: calibrate Grad-CAM localization against the accepted FunnyBird controlled-swap outcome"
echo "MODEL: accepted Koh Joint ResNet-50 Standard CBM, seed 1"
echo "INPUT: accepted 5,000 fixed renderer swaps; exact changed-pixel regions derived from paired RGBs"
echo "OUTPUT: $OUT"
echo "TRAINING: no"
echo "SLURM: no"

test -s "$MODEL" || { echo "ERROR: missing checkpoint $MODEL" >&2; exit 2; }
test -s "$SWAPS" || { echo "ERROR: missing swap CSV $SWAPS" >&2; exit 2; }
test -s "$FUNNYBIRDS_ROOT/parts.json" || { echo "ERROR: missing FunnyBird schema $FUNNYBIRDS_ROOT/parts.json" >&2; exit 2; }

python analysis/test_funnybird_gradcam_swap_calibration.py
if [[ -s "$OUT/SUCCESS.json" ]]; then
  echo "[REUSE COMPLETE] $OUT/SUCCESS.json"
else
  if [[ -e "$OUT" ]]; then
    echo "ERROR: incomplete calibration directory already exists: $OUT" >&2
    echo "Inspect it; do not mix a retry with partial outputs." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$OUT")"
  TMP="$(mktemp -d "${OUT}.tmp.XXXXXX")"
  echo "temporary_output=$TMP"
  python analysis/funnybird_gradcam_swap_calibration.py \
    --funnybirds-root "$FUNNYBIRDS_ROOT" \
    --checkpoint "$MODEL" \
    --swaps "$SWAPS" \
    --out-dir "$TMP" \
    --rows-per-part 100
  test -s "$TMP/SUCCESS.json" || { echo "ERROR: calibration ended without SUCCESS.json; retained $TMP" >&2; exit 2; }
  mv "$TMP" "$OUT"
  echo "[ATOMIC OUTPUT COMMIT] $OUT"
fi
