#!/usr/bin/env bash
# One visible, read-only entry point for the focused FunnyBird follow-ups.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUN_ID="${FOLLOWUP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$REPO" rev-parse --short HEAD)}"
OUTPUT="$CURATED_DATA/funnybird_followup_v3/$RUN_ID"

echo "===== FUNNYBIRD STANDARD-CBM READ-ONLY FOLLOW-UPS ====="
echo "goal=explain candidate contributors and test whether raw-score fingerprints are used"
echo "model=accepted seed-1 Koh Joint ResNet-50 Standard CBM"
echo "inputs=500 ordinary images + 5,000 accepted controlled swaps + Standard/RLv2 labels"
echo "outputs=four PNG figures + source tables + SUCCESS.json"
echo "training=no"
echo "rendering=no"
echo "slurm=no"
echo "output=$OUTPUT"

python "$REPO/curated/analysis/canonical_manifest.py" verify \
  --manifest "$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/SUCCESS.json"
python "$REPO/curated/analysis/canonical_manifest.py" verify \
  --manifest "$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1/SUCCESS.json"

python "$REPO/curated/analysis/funnybird_followup_diagnostics.py" \
  --curated-data "$CURATED_DATA" \
  --output "$OUTPUT"

echo "Completed read-only follow-ups: $OUTPUT"

