#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CURATED_DATA:?set CURATED_DATA to the curated_data directory}"
export PYTHONUNBUFFERED=1
echo "GOAL: test what MCBM compression changed, whether grounding improved, and which loss to investigate next"
echo "WORK: diagnostics and frozen GPU inference only; no scientific training and no Slurm submissions"
echo "[0/8] Check all real inputs and the executed Standard visual baseline before diagnostic fits"
python analysis/mcbm_loss_report.py

KOH_MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1"
KOH_SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
MCBM_SWAP_ROOT="$CURATED_DATA/swap_fixed_v2_attempt2"
VISIBILITY_ROOT="$CURATED_DATA/funnybird_visibility_correction_v1"

echo "[1/8] Verify the accepted Koh Standard model and swap manifests"
python analysis/canonical_manifest.py verify --manifest "$KOH_MODEL_ROOT/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$KOH_SWAP_ROOT/SUCCESS.json"

echo "[2/8] Revalidate the accepted Koh and MCBM fixed-render files"
python analysis/validate_fixed_swaps.py --out "$KOH_SWAP_ROOT"
python analysis/validate_fixed_swaps.py --out "$MCBM_SWAP_ROOT"

echo "[3/8] Derive the same corrected bilateral visibility used by notebooks 02 and 02rl"
python analysis/derive_funnybird_visibility.py \
  --swap-root "$KOH_SWAP_ROOT" \
  --output "$VISIBILITY_ROOT"

echo "[4/8] Prepare all six frozen MCBM swap replays with live progress"
echo "       (adopts previously accepted caches; audit batch 7, defect 5 mitigation)"
python analysis/mcbm_loss_report.py --prepare-replay --gamma all

echo "[5/8] Rebuild notebook 03 from its versioned builder"
python analysis/build_03_standard_mcbm_report.py

# This runner consumes completed checkpoints and validated fixed renders only.
# It does not submit, release, or cancel Slurm jobs.
echo "[6/8] Execute notebook 03; read-only analysis, no Slurm and no training"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/03_funnybirds_mcbm.ipynb

echo "[7/8] Export standalone HTML"
jupyter nbconvert --to html notebooks/03_funnybirds_mcbm.ipynb

echo "[8/8] Restore and verify figure alternative text"
python analysis/repair_nbconvert_alt_text.py \
  notebooks/03_funnybirds_mcbm.ipynb \
  notebooks/03_funnybirds_mcbm.html

echo "Executed standard MCBM report: $ROOT/notebooks/03_funnybirds_mcbm.html"
