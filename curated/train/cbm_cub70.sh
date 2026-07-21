#!/usr/bin/env bash
# Same-backbone CBM on the first 70 CUB classes, using original training labels.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKLS="$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original"
[ -f "$PKLS/train.pkl" ] || python3 "$HERE/../data/cub70/prepare_cub70_pkls.py"
export CUB_PKLS="$PKLS" MODELS="cbm" SEEDS="${SEEDS:-1 2 3}"
bash "$HERE/run_baselines.sh" cub70
