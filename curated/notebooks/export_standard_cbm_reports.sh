#!/usr/bin/env bash
set -euo pipefail

# Export the already-executed standard-CBM notebooks.  This script does not
# execute any cell and does not use a GPU.  Notebook PNG outputs are embedded
# directly in each standalone HTML file.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
ARTIFACT_DIR="$HOME/cbm_artifacts"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ARTIFACT_DIR/standard_cbm_reports_${STAMP}.tar.gz"

mkdir -p "$ARTIFACT_DIR"
cd "$CURATED"

jupyter nbconvert --to html notebooks/02_funnybirds_cbm.ipynb
jupyter nbconvert --to html notebooks/05_cub_cbm.ipynb

python analysis/repair_nbconvert_alt_text.py \
  notebooks/02_funnybirds_cbm.ipynb \
  notebooks/02_funnybirds_cbm.html
python analysis/repair_nbconvert_alt_text.py \
  notebooks/05_cub_cbm.ipynb \
  notebooks/05_cub_cbm.html

tar -czf "$ARCHIVE" \
  -C "$CURATED" \
  notebooks/02_funnybirds_cbm.html \
  notebooks/05_cub_cbm.html \
  notebooks/02_funnybirds_cbm.ipynb \
  notebooks/05_cub_cbm.ipynb

echo "FunnyBird report: $CURATED/notebooks/02_funnybirds_cbm.html"
echo "CUB70 report:      $CURATED/notebooks/05_cub_cbm.html"
echo "Download archive: $ARCHIVE"
