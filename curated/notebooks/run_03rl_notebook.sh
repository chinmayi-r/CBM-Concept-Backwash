#!/usr/bin/env bash
# Execute the exact Notebook-03 analysis template on the accepted RLv2 MCBM sweep.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

MODEL_PREFIX=funnybirds-mcbm-rlv2matched
SWAP_ROOT_NAME=swap_fixed_v3_matched
TABLE_ROOT="$CURATED_DATA/mcbm_notebook03rl_tables"
PATHWAY_ROOT="$CURATED_DATA/mcbm_swap_pathway_rlv2_v3"

echo "GOAL: repeat every Notebook-03 MCBM question after visibility-aware relabeling"
echo "MODELS: accepted seed-1 RLv2 MCBM gamma 0, 0.1, 0.3, 1, 3, 5"
echo "COMPARISON: exact executed Standard-MCBM output, then the same RLv2 calculation"
echo "WORK: read-only diagnostics and frozen inference; no training and no Slurm"

echo "[0/9] Run synthetic/source tests before touching real artifacts"
python analysis/test_mcbm_loss_report.py
python analysis/test_mcbm_swap_pathway_report.py
python analysis/test_03rl_parity_builder.py
python analysis/test_validate_fixed_swaps.py

echo "[1/9] Verify all six RLv2 checkpoints, exports, configs, and fixed-swap CSVs"
python analysis/mcbm_loss_report.py \
  --model-prefix "$MODEL_PREFIX" \
  --swap-root-name "$SWAP_ROOT_NAME" \
  --replay-root-name mcbm_notebook03rl_replay \
  --skip-standard-notebook-check

echo "[2/9] Revalidate identical RGB model inputs; disclose auxiliary part-map metadata"
echo "The model reads RGB, not the part-map PNG. Notebook visibility comes from the separately verified canonical visibility table."
python analysis/validate_fixed_swaps.py \
  --out "$CURATED_DATA/$SWAP_ROOT_NAME" \
  --part-map-policy disclose

echo "[3/9] Prepare/resume the same fully captioned source tables for every gamma"
python analysis/prepare_mcbm_report_tables.py \
  --model-prefix "$MODEL_PREFIX" \
  --swap-root-name "$SWAP_ROOT_NAME" \
  --output "$TABLE_ROOT"

echo "[4/9] Prepare/reuse counterfactual h for the accepted RLv2 swaps"
python analysis/mcbm_loss_report.py \
  --prepare-replay --gamma all --disable-tf32 \
  --model-prefix "$MODEL_PREFIX" \
  --swap-root-name "$SWAP_ROOT_NAME" \
  --replay-root-name mcbm_notebook03rl_replay \
  --skip-standard-notebook-check

echo "[5/9] Rebuild calibrated h -> q(h)=z pathway and frozen-head intervention"
python analysis/mcbm_swap_pathway_report.py \
  --model-prefix "$MODEL_PREFIX" \
  --swap-root-name "$SWAP_ROOT_NAME" \
  --counterfactual-replay-root-name mcbm_notebook03rl_replay \
  --original-replay-root-name mcbm_notebook03rl_original_replay \
  --analysis-version mcbm_swap_pathway_rlv2_v3 \
  --output "$PATHWAY_ROOT"

echo "[6/9] Build Notebook 03rl from the exact current Notebook-03 template"
python analysis/build_03rl_notebook.py

echo "[7/9] Execute Notebook 03rl; no scientific training"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/03rl_funnybirds_mcbm_relabeled.ipynb

echo "[8/9] Export HTML and restore figure alternative text"
jupyter nbconvert --to html notebooks/03rl_funnybirds_mcbm_relabeled.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/03rl_funnybirds_mcbm_relabeled.ipynb \
  notebooks/03rl_funnybirds_mcbm_relabeled.html

echo "[9/9] Complete"
echo "Executed notebook: $CURATED/notebooks/03rl_funnybirds_mcbm_relabeled.ipynb"
echo "Rendered report:   $CURATED/notebooks/03rl_funnybirds_mcbm_relabeled.html"
