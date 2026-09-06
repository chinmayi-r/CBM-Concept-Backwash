#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CURATED_DATA:?set CURATED_DATA to the curated_data directory}"

KOH_MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1"
KOH_SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
MCBM_SWAP_ROOT="$CURATED_DATA/swap_fixed_v2_attempt2"
VISIBILITY_ROOT="$CURATED_DATA/funnybird_visibility_correction_v1"

echo "[1/7] Verify the accepted Koh Standard model and swap manifests"
python analysis/canonical_manifest.py verify --manifest "$KOH_MODEL_ROOT/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$KOH_SWAP_ROOT/SUCCESS.json"

echo "[2/7] Revalidate the accepted Koh and MCBM fixed-render files"
python analysis/validate_fixed_swaps.py --out "$KOH_SWAP_ROOT"
python analysis/validate_fixed_swaps.py --out "$MCBM_SWAP_ROOT"

echo "[3/7] Derive the same corrected bilateral visibility used by notebooks 02 and 02rl"
python analysis/derive_funnybird_visibility.py \
  --swap-root "$KOH_SWAP_ROOT" \
  --output "$VISIBILITY_ROOT"

echo "[4/7] Rebuild notebook 03 from its versioned builder"
python analysis/build_03_standard_mcbm_report.py

# This runner consumes completed checkpoints and validated fixed renders only.
# It does not submit, release, or cancel Slurm jobs.
echo "[5/7] Execute notebook 03; read-only analysis, no Slurm and no training"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/03_funnybirds_mcbm.ipynb

echo "[6/7] Export standalone HTML"
jupyter nbconvert --to html notebooks/03_funnybirds_mcbm.ipynb

echo "[7/7] Restore and verify figure alternative text"
python analysis/repair_nbconvert_alt_text.py \
  notebooks/03_funnybirds_mcbm.ipynb \
  notebooks/03_funnybirds_mcbm.html

echo "Executed standard MCBM report: $ROOT/notebooks/03_funnybirds_mcbm.html"
