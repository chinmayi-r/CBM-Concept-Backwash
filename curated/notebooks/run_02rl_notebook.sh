#!/usr/bin/env bash
# Build, execute, and export the CBM-only matched RLv2 causal notebook.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MODEL_BASE="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
VISIBILITY_ROOT="$CURATED_DATA/funnybird_visibility_correction_v1"

cd "$CURATED"

echo "[1/7] Verify accepted Standard, RLv2, and matched-swap manifests"
python analysis/canonical_manifest.py verify --manifest "$MODEL_BASE/standard/seed1/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$MODEL_BASE/rlv2/seed1/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$SWAP_ROOT/SUCCESS.json"

echo "[2/7] Revalidate the identical renderer-swap files"
python analysis/validate_fixed_swaps.py --out "$SWAP_ROOT"

echo "[3/7] Derive corrected bilateral visibility; no inference or training"
python analysis/derive_funnybird_visibility.py --swap-root "$SWAP_ROOT" --output "$VISIBILITY_ROOT"

echo "[4/7] Require the current executed Standard notebook as the visual baseline"
test -s notebooks/02_funnybirds_cbm.ipynb

echo "[5/7] Build notebook 02RL from accepted Koh artifacts"
python analysis/build_02rl_notebook.py

echo "[6/7] Execute notebook 02RL; read-only analysis, no Slurm and no training"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/02rl_funnybirds_cbm_relabeled.ipynb

echo "[7/7] Export standalone HTML and restore figure alt text"
jupyter nbconvert --to html notebooks/02rl_funnybirds_cbm_relabeled.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/02rl_funnybirds_cbm_relabeled.ipynb \
  notebooks/02rl_funnybirds_cbm_relabeled.html

echo "Executed notebook: $CURATED/notebooks/02rl_funnybirds_cbm_relabeled.ipynb"
echo "Rendered report:   $CURATED/notebooks/02rl_funnybirds_cbm_relabeled.html"
