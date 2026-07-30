#!/usr/bin/env bash
# Standard CBM trained with FunnyBirds image-level visibility labels.
# Same minimal_cbm trainer/backbone as funnybirds-cbm; only the concept labels differ.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEEDS="${SEEDS:-1 2 3}"
RL_PKLS="$CURATED_DATA/funnybirds_processed_rl"
[ -f "$RL_PKLS/train.pkl" ] || {
  echo "ERROR: missing $RL_PKLS/train.pkl; build the image-level labels first" >&2; exit 1; }
RL_TRAINVAL="${RL_PKLS}_trainval"
[ -f "$RL_TRAINVAL/train.pkl" ] && [ -f "$RL_TRAINVAL/test.pkl" ] || {
  echo "ERROR: missing matched RLv2 train/validation split: $RL_TRAINVAL" >&2
  echo "Build it with the same deterministic split used for standard labels:" >&2
  echo "  python data/make_val_split.py --pkls-dir \"$RL_PKLS\" --seed 42" >&2
  exit 1
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"
source "$HERE/_paths.sh"
python3 "$CURATED/analysis/audit_03rl_accuracy.py" \
  --curated-data "$CURATED_DATA"
export FB_PKLS="$RL_TRAINVAL"
GEN_DIR="$MCBM/configs/funnybirds"; mkdir -p "$GEN_DIR"
RL_TAG="${RL_TAG:-rlv2matched}"
BASE="funnybirds-cbm-${RL_TAG}"
CFG="$GEN_DIR/${BASE}.yaml"
gen_config "$HERE/configs/funnybirds-cbm.yaml" "$CFG" funnybirds "${ARCH:-resnet50}" ""
grep -q "funnybirds_processed_rl_trainval" "$CFG" || {
  echo "ERROR: matched relabeled train/validation path missing from $CFG" >&2; exit 1; }
for seed in $SEEDS; do
  echo ">>> CBM-RL seed=$seed"
  (cd "$HERE" && python3 run_mcbm.py "$BASE" -s "$seed")
done
echo "Done -> $MCBM/results/$BASE/<seed>"
