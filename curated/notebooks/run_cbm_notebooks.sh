#!/usr/bin/env bash
# Execute and export the complete CBM notebook path in narrative order.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${CUB_ROOT:?export CUB_ROOT}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

required=(
  "$CURATED_DATA/funnybirds_processed/test.pkl"
  "$CURATED_DATA/swap/funnybirds-cbm-s1.csv"
  "$CURATED_DATA/grounding/funnybirds-cbm-s1.parquet"
  "$CURATED_DATA/species_probe/funnybirds-cbm-s1.json"
  "$CURATED_DATA/CUB_processed/class_attr_data_10/test.pkl"
  "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/test.pkl"
  "$CURATED_DATA/cub70_visibility.parquet"
  "$CURATED_DATA/cub70_relabel_diagnostics.parquet"
  "$CURATED_DATA/cub70_eval/cub-cbm-s1.parquet"
  "$CURATED_DATA/cub70_eval/cub70-cbm-s1.parquet"
)
missing=0
for path in "${required[@]}"; do
  if [ ! -s "$path" ]; then
    echo "MISSING: $path" >&2
    missing=1
  fi
done
if [ "$missing" = 1 ]; then
  echo "Stop: produce the missing artifacts before exporting final notebooks." >&2
  exit 1
fi

for name in \
  01_funnybirds_analysis \
  02_funnybirds_cbm \
  04_cub_analysis \
  05_cub_cbm
do
  echo "### execute $name"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "notebooks/${name}.ipynb"
  echo "### export $name.html"
  jupyter nbconvert --to html "notebooks/${name}.ipynb"
done

echo "All CBM notebooks executed with complete inputs."
