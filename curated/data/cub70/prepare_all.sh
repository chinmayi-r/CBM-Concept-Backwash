#!/usr/bin/env bash
# One checked entry point for every CUB/CUB70 data artifact needed by notebooks
# 04-06 and the CUB/CUB70 CBM training jobs.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${CUB_ROOT:?export CUB_ROOT to the raw CUB_200_2011 directory}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/../.." && pwd)"
CUB_ATTR_FILE="${CUB_ATTR_FILE:-$(dirname "$CUB_ROOT")/attributes.txt}"

echo "### [1/4] deterministic official CUB pickles + 112 concepts + CUB70 filter"
python3 "$CURATED/data/cub/prepare_cub_data.py" \
  --cub-root "$CUB_ROOT" \
  --data-root "$CURATED_DATA" \
  --attr-names "$CUB_ATTR_FILE" \
  "$@"

echo "### [2/4] CUB70 part-mask visibility"
python3 "$HERE/build_cub70_visibility.py" --data-root "$CURATED_DATA"

echo "### [3/4] visibility-aware evaluation labels (test-only diagnostic)"
python3 "$HERE/relabel_cub_with_cub70.py" \
  --data-root "$CURATED_DATA" \
  --data-dir "CUB_processed/class_attr_data_10_cub70_original" \
  --cub-root "$CUB_ROOT" \
  --attr-names "$CUB_ATTR_FILE"

echo "### [4/4] required outputs"
for path in \
  "$CURATED_DATA/CUB_processed/class_attr_data_10/train.pkl" \
  "$CURATED_DATA/CUB_processed/class_attr_data_10/test.pkl" \
  "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/train.pkl" \
  "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/test.pkl" \
  "$CURATED_DATA/CUB_processed/attributes.txt" \
  "$CURATED_DATA/cub70_visibility.parquet" \
  "$CURATED_DATA/cub70_relabel_diagnostics.parquet"
do
  test -s "$path" || { echo "ERROR: missing or empty $path" >&2; exit 1; }
  ls -lh "$path"
done
echo "CUB/CUB70 preparation complete."
