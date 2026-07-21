#!/usr/bin/env bash
# Same-backbone MCBM gamma sweep on first 70 CUB classes, original labels.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKLS="$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original"
[ -f "$PKLS/train.pkl" ] || python3 "$HERE/../data/cub70/prepare_cub70_pkls.py"
export CUB_PKLS="$PKLS"
bash "$HERE/mcbm_gamma_sweep.sh" cub70
