#!/usr/bin/env bash
#SBATCH --job-name=fb_xpart_pre
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail

: "${REPO:?set REPO}"
: "${CURATED_DATA:?set CURATED_DATA}"
: "${OUT:?set OUT}"

CONDA_ENV="${CONDA_ENV:-cubvision-gpu}"
ROWS_PER_INPUT_PART="${ROWS_PER_INPUT_PART:-18}"
REGIME_LABEL="${REGIME_LABEL:-Standard}"
CSV_NAME="${CSV_NAME:-funnybirds-cbm-s1.csv}"
SELECTION_CSV="${SELECTION_CSV:-}"
CHECKPOINT="${CHECKPOINT:-$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/final_model_1.pth}"
MODEL_MANIFEST="${MODEL_MANIFEST:-$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/SUCCESS.json}"
SWAP_ROOT="${SWAP_ROOT:-$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1}"
FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-$CURATED_DATA/FunnyBirds}"

echo "GOAL: find intended and unexpected pathways from every replaced part to every concept block"
echo "WORK: frozen accepted-model replay on a small balanced subset; no training and no fitted probe"
echo "MODEL: accepted seed-1 ResNet-50 Koh-architecture Joint CBM, $REGIME_LABEL labels"
echo "INPUT PARTS: tail wing beak foot eye"
echo "OUTPUT BLOCKS: tail wing beak foot eye"
if [ -n "$SELECTION_CSV" ]; then
  echo "ROWS: exact accepted render identities from $SELECTION_CSV"
else
  echo "ROWS: $ROWS_PER_INPUT_PART accepted controlled swaps per input part"
fi
echo "QUANTITIES: raw block movement; unchanged-block correctness damage; saved-head source/donor gap contribution; intended diagonal response"
echo "CHECKPOINT: $CHECKPOINT"
echo "SWAPS: $SWAP_ROOT/$CSV_NAME"
echo "OUTPUT: $OUT"
echo "STAGES: behavioral test -> manifest/hash gates -> real schema/count preflight -> frozen replay -> 5x5 tables/figure"

cd "$REPO"
module load anaconda3/2025.6 2>/dev/null || true
set +u
conda activate "$CONDA_ENV"
set -u
export WANDB_MODE=offline WANDB_DISABLED=true
ulimit -c 0 || true

python curated/analysis/test_funnybird_allpart_swap_preflight.py
test -s "$CHECKPOINT"
test -s "$FUNNYBIRDS_ROOT/parts.json"
test -s "$FUNNYBIRDS_ROOT/dataset_test.json"
test -s "$SWAP_ROOT/$CSV_NAME"
test ! -e "$OUT" || { echo "ERROR: output already exists: $OUT"; exit 2; }

python curated/analysis/canonical_manifest.py verify \
  --manifest "$MODEL_MANIFEST"
python curated/analysis/canonical_manifest.py verify \
  --manifest "$SWAP_ROOT/SUCCESS.json"
python curated/analysis/validate_fixed_swaps.py --out "$SWAP_ROOT"

selection_args=()
if [ -n "$SELECTION_CSV" ]; then
  test -s "$SELECTION_CSV"
  selection_args=(--selection-csv "$SELECTION_CSV")
fi
python curated/analysis/funnybird_allpart_swap_preflight.py \
  --funnybirds-root "$FUNNYBIRDS_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --swap-root "$SWAP_ROOT" \
  --csv-name "$CSV_NAME" \
  --regime-label "$REGIME_LABEL" \
  --out "$OUT" \
  --parts tail wing beak foot eye \
  --rows-per-input-part "$ROWS_PER_INPUT_PART" \
  "${selection_args[@]}"

test -s "$OUT/PREFLIGHT_STATUS.json"
python - "$OUT/PREFLIGHT_STATUS.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("PREFLIGHT STATUS:", data["status"])
print("scientific_result:", data["scientific_result"])
print("selected_swap_rows:", data["selected_swap_rows"])
print("pathway_rows:", data["pathway_rows"])
print("max_absolute_score_replay_error:", data["max_absolute_score_replay_error"])
print("outcome_disagreements:", data["outcome_disagreements"])
PY
echo "ACCEPTED FOR SMALL ALL-PART MECHANISM PREFLIGHT: $OUT"
echo "Interpretation waits for review of every matrix cell and the ranked cross-part table."
