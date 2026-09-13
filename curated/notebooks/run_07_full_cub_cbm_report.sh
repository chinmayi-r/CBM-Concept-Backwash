#!/usr/bin/env bash
# Render the official full-CUB Koh Joint report after entry 11 completes.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
CURATED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CURATED"
MODEL="$CURATED_DATA/koh_joint_resnet_decay_continuation_v1/cub/standard/seed1"

echo "GOAL: finish the separate 200-species Full-CUB Standard-CBM chapter"
echo "MODEL: official Koh-architecture Joint ResNet-50 decay continuation, seed 1"
echo "INPUTS: 5,794-image full test export; released masks only on the CUB70 subset"
echo "WORK: frozen spatial/saved-head audit and report rendering"
echo "TRAINING: no (this runner refuses to start until entry 11 already succeeded)"

python analysis/canonical_manifest.py verify --manifest "$MODEL/SUCCESS.json"
test -s "$MODEL/final_test.parquet"
if [[ ! -s "$CURATED_DATA/cub_koh_spatial_v1/full_cub_standard_s1/SUCCESS.json" ]]; then
  bash notebooks/run_cub_koh_spatial_audit.sh full
else
  echo "[REUSE COMPLETE] full-CUB spatial/saved-head audit"
fi
python analysis/build_07_full_cub_cbm_report.py
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/07_full_cub_cbm.ipynb
jupyter nbconvert --to html notebooks/07_full_cub_cbm.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/07_full_cub_cbm.ipynb notebooks/07_full_cub_cbm.html
echo "[FULL-CUB STANDARD REPORT COMPLETE] $CURATED/notebooks/07_full_cub_cbm.html"
