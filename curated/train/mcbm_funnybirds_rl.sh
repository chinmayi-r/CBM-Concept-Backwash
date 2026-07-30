#!/usr/bin/env bash
# Train the RELABELED (image-level) MCBM gamma sweep. Same OFFICIAL trainer, same
# backbone, same gammas as the standard sweep (mcbm_gamma_sweep.sh funnybirds) --
# the ONLY difference is the concept labels: a concept is set to 0 when its part is
# occluded in that image (visibility-aware / image-level), which BREAKS the perfect
# concept<->species correlation. This is the causal test for notebook 03rl:
#   standard (concept from render parameters; placeholder already means absent)
#   vs relabeled (also zero non-placeholder parts with negligible visible area).
#
# Prereq -- build the relabeled pkls into a SEPARATE dir (won't touch the standard ones):
#   python data/funnybirds/build_funnybirds_cbm_data.py \
#       --funnybirds-root "$FB" --labels image_level --out-name funnybirds_processed_rl
#
# Then:
#   GAMMAS="0 0.1 0.3 1 3 5" SEEDS="1" bash train/mcbm_funnybirds_rl.sh
#
# Results land in results/funnybirds-mcbm-rl-g<tag>/<seed>/ -> distinct from standard.
# Swap:  CONFIG_PREFIX=funnybirds-mcbm-rl sbatch train/renderer_swap.slurm
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
GAMMAS="${GAMMAS:-0 0.1 0.3 1 3 5}"
SEEDS="${SEEDS:-1}"
RL_TAG="${RL_TAG:-rlv2matched}"
RL_PREFIX="funnybirds-mcbm-${RL_TAG}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # curated/train
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"

# point the config generator at the RELABELED pkls (gen_config -> _pick_pkls honors FB_PKLS)
RL_PKLS="$CURATED_DATA/funnybirds_processed_rl"
if [ ! -f "$RL_PKLS/train.pkl" ]; then
  echo "[ERROR] no $RL_PKLS/train.pkl" >&2
  echo "  build it first:" >&2
  echo "  python data/funnybirds/build_funnybirds_cbm_data.py --funnybirds-root \"\$FB\" \\" >&2
  echo "      --labels image_level --out-name funnybirds_processed_rl" >&2
  exit 1
fi
RL_TRAINVAL="${RL_PKLS}_trainval"
if [ ! -f "$RL_TRAINVAL/train.pkl" ] || [ ! -f "$RL_TRAINVAL/test.pkl" ]; then
  echo "[ERROR] no matched RLv2 train/validation split at $RL_TRAINVAL" >&2
  echo "  build it with the same deterministic split used for standard labels:" >&2
  echo "  python data/make_val_split.py --pkls-dir \"$RL_PKLS\" --seed 42" >&2
  exit 1
fi
python3 "$CURATED/analysis/audit_03rl_accuracy.py" \
  --curated-data "$CURATED_DATA"

GEN_DIR="$MCBM/configs/funnybirds"; mkdir -p "$GEN_DIR"
python3 "$CURATED/analysis/prepare_03rl_matched_configs.py" \
  --curated-data "$CURATED_DATA" --rl-tag "$RL_TAG" --gammas $GAMMAS
python3 "$CURATED/analysis/audit_03rl_accuracy.py" \
  --curated-data "$CURATED_DATA" --configs --skip-cbm \
  --rl-tag "$RL_TAG" --gammas $GAMMAS
echo "### MCBM-RL sweep  run=$RL_PREFIX  exact-standard-config-copy=yes  gammas=[$GAMMAS]  seeds=[$SEEDS]"
echo "    pkls=$RL_TRAINVAL  gen=$GEN_DIR"
for g in $GAMMAS; do
  gtag="${g//./p}"                                          # 0.1 -> 0p1
  base="${RL_PREFIX}-g${gtag}"                              # prefix (config dir) == funnybirds
  for s in $SEEDS; do
    echo ">>> RL gamma=$g seed=$s  ($base)"
    ( cd "$HERE" && python3 run_mcbm.py "$base" -s "$s" )
  done
done
echo "Done. RL results in $MCBM/results/${RL_PREFIX}-g*/<seed>/."
echo "Next: CONFIG_PREFIX=$RL_PREFIX sbatch train/renderer_swap.slurm"
