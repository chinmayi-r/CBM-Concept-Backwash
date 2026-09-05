#!/usr/bin/env bash
# Build, execute, and export the CBM-only matched RLv2 causal notebook.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MODEL_BASE="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
VISIBILITY_ROOT="$CURATED_DATA/funnybird_visibility_correction_v1"
# Key expensive replay outputs by the analysis implementation itself. A later
# notebook-only render commit therefore reuses them instead of replaying 3,040
# CUDA images twice for no scientific reason.
ANALYSIS_KEY="$(git hash-object "$CURATED/analysis/funnybird_followup_diagnostics.py" | cut -c1-12)"
MATCHED_FOLLOWUP_ROOT="$CURATED_DATA/funnybird_followup_matched_v1/$ANALYSIS_KEY"

cd "$CURATED"

echo "[1/9] Verify accepted Standard, RLv2, and matched-swap manifests"
python analysis/canonical_manifest.py verify --manifest "$MODEL_BASE/standard/seed1/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$MODEL_BASE/rlv2/seed1/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$SWAP_ROOT/SUCCESS.json"

echo "[2/9] Revalidate the identical renderer-swap files"
python analysis/validate_fixed_swaps.py --out "$SWAP_ROOT"

echo "[3/9] Derive corrected bilateral visibility; no inference or training"
python analysis/derive_funnybird_visibility.py --swap-root "$SWAP_ROOT" --output "$VISIBILITY_ROOT"

echo "[4/9] Require the current executed Standard notebook as the visual baseline"
test -s notebooks/02_funnybirds_cbm.ipynb

echo "[5/9] Run the exact same read-only follow-ups for Standard"
if [[ ! -s "$MATCHED_FOLLOWUP_ROOT/standard/SUCCESS.json" ]]; then
  if [[ -e "$MATCHED_FOLLOWUP_ROOT/standard" ]]; then
    echo "ERROR: incomplete Standard follow-up directory already exists: $MATCHED_FOLLOWUP_ROOT/standard" >&2
    exit 1
  fi
  python analysis/funnybird_followup_diagnostics.py \
    --regime standard \
    --output "$MATCHED_FOLLOWUP_ROOT/standard"
else
  echo "[REUSE COMPLETE] $MATCHED_FOLLOWUP_ROOT/standard/SUCCESS.json"
fi

echo "[6/9] Run the same read-only follow-ups for RLv2; only regime/model/CSV changes"
if [[ ! -s "$MATCHED_FOLLOWUP_ROOT/rlv2/SUCCESS.json" ]]; then
  if [[ -e "$MATCHED_FOLLOWUP_ROOT/rlv2" ]]; then
    echo "ERROR: incomplete RLv2 follow-up directory already exists: $MATCHED_FOLLOWUP_ROOT/rlv2" >&2
    exit 1
  fi
  python analysis/funnybird_followup_diagnostics.py \
    --regime rlv2 \
    --output "$MATCHED_FOLLOWUP_ROOT/rlv2"
else
  echo "[REUSE COMPLETE] $MATCHED_FOLLOWUP_ROOT/rlv2/SUCCESS.json"
fi
export FUNNYBIRD_MATCHED_FOLLOWUP_ROOT="$MATCHED_FOLLOWUP_ROOT"

echo "[7/9] Build notebook 02RL from accepted Koh artifacts"
python analysis/build_02rl_notebook.py

echo "[8/9] Execute notebook 02RL; read-only analysis, no Slurm and no training"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/02rl_funnybirds_cbm_relabeled.ipynb

echo "[9/9] Export standalone HTML and restore figure alt text"
jupyter nbconvert --to html notebooks/02rl_funnybirds_cbm_relabeled.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/02rl_funnybirds_cbm_relabeled.ipynb \
  notebooks/02rl_funnybirds_cbm_relabeled.html

echo "Executed notebook: $CURATED/notebooks/02rl_funnybirds_cbm_relabeled.ipynb"
echo "Rendered report:   $CURATED/notebooks/02rl_funnybirds_cbm_relabeled.html"
