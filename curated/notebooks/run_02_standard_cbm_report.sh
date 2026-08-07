#!/usr/bin/env bash
# Rebuild and execute only the standard FunnyBird CBM report.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

python analysis/build_standard_cbm_reports.py --only 02
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/02_funnybirds_cbm.ipynb
python analysis/finalize_standard_cbm_reports.py \
  notebooks/02_funnybirds_cbm.ipynb
jupyter nbconvert --to html notebooks/02_funnybirds_cbm.ipynb

echo "Executed standard FunnyBird CBM report: $CURATED/notebooks/02_funnybirds_cbm.html"
