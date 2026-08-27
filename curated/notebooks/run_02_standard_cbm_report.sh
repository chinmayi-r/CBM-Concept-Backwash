#!/usr/bin/env bash
# Rebuild, execute, and export only the standard FunnyBird CBM report.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"

cd "$CURATED"

echo "[1/5] Verify final Koh Standard and fixed-swap manifests"
python analysis/canonical_manifest.py verify --manifest "$MODEL_ROOT/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$SWAP_ROOT/SUCCESS.json"

echo "[2/5] Revalidate the accepted renderer-swap files"
python analysis/validate_fixed_swaps.py --out "$SWAP_ROOT"

echo "[3/5] Rebuild notebook 02 from its versioned builder"
python analysis/build_standard_cbm_reports.py --only 02

echo "[4/5] Execute notebook 02 in place; this reads completed artifacts and submits no jobs"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/02_funnybirds_cbm.ipynb

echo "[5/5] Export standalone HTML and restore figure alt text"
jupyter nbconvert --to html notebooks/02_funnybirds_cbm.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/02_funnybirds_cbm.ipynb \
  notebooks/02_funnybirds_cbm.html

echo "Executed notebook: $CURATED/notebooks/02_funnybirds_cbm.ipynb"
echo "Rendered report:   $CURATED/notebooks/02_funnybirds_cbm.html"
