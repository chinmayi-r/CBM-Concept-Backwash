#!/usr/bin/env bash
# Run the four predeclared read-only notebook-02 follow-up diagnostics
# (DECISIONS.md section D.6). No training, no rendering; reads accepted
# seed-1 artifacts and writes tables to $CURATED_DATA/diagnostics_predeclared_v1.
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

echo "[2/5] D6.1 dimension-adjusted conditional species information"
python analysis/diag_dimension_adjusted_information.py

echo "[3/5] D6.2 part-profile transfer (full-block and off-target)"
python analysis/diag_profile_transfer.py

echo "[4/5] D6.3 conflict -> response components"
python analysis/diag_conflict_components.py

echo "[5/5] D6.4 grouped continuous risk model"
python analysis/diag_grouped_risk_model.py

echo "Tables written under: $CURATED_DATA/diagnostics_predeclared_v1"
