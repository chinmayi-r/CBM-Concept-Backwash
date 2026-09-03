#!/usr/bin/env bash
# Run the five prospectively specified read-only notebook-02 follow-up diagnostics
# (DECISIONS.md section D.6). No training, no rendering; reads accepted
# seed-1 artifacts and writes tables to $CURATED_DATA/diagnostics_predeclared_v2.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
RUN_ID="${DIAGNOSTICS_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$CURATED/.." rev-parse --short HEAD)}"
OUTPUT_ROOT="$CURATED_DATA/diagnostics_predeclared_v2/$RUN_ID"
export DIAGNOSTICS_OUTPUT_DIR="$OUTPUT_ROOT"

cd "$CURATED"

mkdir -p "$OUTPUT_ROOT"
if [[ -e "$OUTPUT_ROOT/SUCCESS.json" ]]; then
  echo "ERROR: this diagnostic run id already completed: $RUN_ID" >&2
  exit 2
fi

echo "===== READ-ONLY D6 FOLLOW-UP ====="
echo "goal=explain remaining FunnyBird controlled-swap variation without causal percentages"
echo "framework=accepted Koh Joint ResNet-50 Standard CBM"
echo "dataset=FunnyBird fixed-render swaps seed=1"
echo "actions=verify inputs -> five diagnostics -> atomic success manifest"
echo "training=no rendering=no slurm=no"
echo "output=$OUTPUT_ROOT"
echo "commit=$(git -C .. rev-parse HEAD)"

echo "[1/7] Verify final Koh Standard and fixed-swap manifests"
python analysis/canonical_manifest.py verify --manifest "$MODEL_ROOT/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$SWAP_ROOT/SUCCESS.json"

echo "[2/7] D6.1 conditional species information beyond binary labels"
python analysis/diag_dimension_adjusted_information.py

echo "[3/7] D6.2 donor/source/neither part-profile transfer"
python analysis/diag_profile_transfer.py

echo "[4/7] D6.3 conflict -> matched response components"
python analysis/diag_conflict_components.py

echo "[5/7] D6.4 genuinely nested grouped continuous risk model"
python analysis/diag_grouped_risk_model.py

echo "[6/7] D6.5 unchanged saved-head use of within-label magnitudes"
python analysis/diag_saved_head_use.py

echo "[7/7] Verify every table and write provenance manifest"
python analysis/finalize_predeclared_diagnostics.py \
  --output "$OUTPUT_ROOT" \
  --model-root "$MODEL_ROOT" \
  --swap-root "$SWAP_ROOT"

echo "Accepted tables and SUCCESS.json written under: $OUTPUT_ROOT"
