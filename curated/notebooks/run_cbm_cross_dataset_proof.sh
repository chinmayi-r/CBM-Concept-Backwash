#!/usr/bin/env bash
# Execute the primary standard-CBM comparison: FunnyBird notebook 02, then CUB70 notebook 05.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

python analysis/build_standard_cbm_reports.py

# Refresh CUB exports. Older files put the encoder's latent slot in the
# schema's raw-logit column. This only replays saved concept heads: no image
# inference, GPU work, or training.
mkdir -p "$CURATED_DATA/cub70_eval"
python analysis/cub70_export_eval.py \
  --config cub70-cbm --seed 1 --epoch 100 \
  --out "$CURATED_DATA/cub70_eval/cub70-cbm-s1.parquet"
python analysis/cub70_export_eval.py \
  --config cub-cbm --seed 1 --epoch 100 \
  --out "$CURATED_DATA/cub70_eval/cub-cbm-s1.parquet"

for name in 02_funnybirds_cbm 05_cub_cbm; do
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "notebooks/${name}.ipynb"
  python analysis/finalize_standard_cbm_reports.py "notebooks/${name}.ipynb"
  jupyter nbconvert --to html "notebooks/${name}.ipynb"
done

echo "Executed the standard-CBM FunnyBird/CUB70 proof comparison."
