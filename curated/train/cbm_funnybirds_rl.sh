#!/usr/bin/env bash
# Standard CBM trained with FunnyBirds image-level visibility labels.
# Same minimal_cbm trainer/backbone as funnybirds-cbm; only the concept labels differ.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEEDS="${SEEDS:-1 2 3}"
RL_PKLS="$CURATED_DATA/funnybirds_processed_rl"
[ -f "$RL_PKLS/train.pkl" ] || {
  echo "ERROR: missing $RL_PKLS/train.pkl; build the image-level labels first" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"
source "$HERE/_paths.sh"
export FB_PKLS="$RL_PKLS"
GEN_DIR="$MCBM/configs/funnybirds"; mkdir -p "$GEN_DIR"
BASE="funnybirds-cbm-rl"
CFG="$GEN_DIR/${BASE}.yaml"
gen_config "$HERE/configs/funnybirds-cbm.yaml" "$CFG" funnybirds "${ARCH:-resnet50}" ""
grep -q "funnybirds_processed_rl" "$CFG" || { echo "ERROR: relabeled path missing from $CFG" >&2; exit 1; }
for seed in $SEEDS; do
  echo ">>> CBM-RL seed=$seed"
  (cd "$HERE" && python3 run_mcbm.py "$BASE" -s "$seed")
done
echo "Done -> $MCBM/results/$BASE/<seed>"
