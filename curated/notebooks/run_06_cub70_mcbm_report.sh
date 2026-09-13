#!/usr/bin/env bash
# Render the finite historical-recipe CUB70 MCBM gamma chapter.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
CURATED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CURATED"

echo "GOAL: compare accepted CUB70 MCBM gamma 0/0.1/0.3/1 with official Koh Standard"
echo "INPUTS: same 1,976 CUB70 test photographs and released masks"
echo "WORK: read-only exports, diagnostics, and notebook rendering"
echo "TRAINING: no"
echo "EXCLUDED: original-recipe gamma 3/5 are errors; stabilized lane is a separate recipe"

python analysis/canonical_manifest.py verify --manifest \
  "$CURATED_DATA/koh_joint_resnet_v1/cub70/standard/seed1/SUCCESS.json"
python analysis/inventory_06_cub_mcbm.py --curated-data "$CURATED_DATA"

for spec in "cub70-mcbm-g0" "cub70-mcbm-g0p1" "cub70-mcbm-g0p3" "cub70-mcbm-g1"; do
  output="$CURATED_DATA/cub70_eval/${spec}-s1.parquet"
  if [[ ! -s "$output" ]]; then
    echo "[EXPORT] $spec seed 1 -> $output"
    python analysis/cub70_export_eval.py --config "$spec" --seed 1 --epoch 100 --out "$output"
  else
    echo "[REUSE COMPLETE] $output"
  fi
done

python analysis/build_06_cub70_mcbm_report.py
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/06_cub_mcbm.ipynb
jupyter nbconvert --to html notebooks/06_cub_mcbm.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/06_cub_mcbm.ipynb notebooks/06_cub_mcbm.html
echo "[CUB70 MCBM REPORT COMPLETE] $CURATED/notebooks/06_cub_mcbm.html"
