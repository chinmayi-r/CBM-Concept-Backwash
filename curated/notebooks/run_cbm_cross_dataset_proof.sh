#!/usr/bin/env bash
# Execute the primary standard-CBM comparison: FunnyBird notebook 02, then CUB70 notebook 05.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

python analysis/build_standard_cbm_reports.py

for name in 02_funnybirds_cbm 05_cub_cbm; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "notebooks/${name}.ipynb"
  python analysis/finalize_standard_cbm_reports.py "notebooks/${name}.ipynb"
  jupyter nbconvert --to html "notebooks/${name}.ipynb"
done

echo "Executed the standard-CBM FunnyBird/CUB70 proof comparison."
