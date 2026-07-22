#!/usr/bin/env bash
# Train the RELABELED (image-level) MCBM gamma sweep. Same OFFICIAL trainer, same
# backbone, same gammas as the standard sweep (mcbm_gamma_sweep.sh funnybirds) --
# the ONLY difference is the concept labels: a concept is set to 0 when its part is
# occluded in that image (visibility-aware / image-level), which BREAKS the perfect
# concept<->species correlation. This is the causal test for notebook 03rl:
#   standard (concept=f(species))  vs  relabeled (concept=f(what's actually visible)).
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
ARCH="${ARCH:-resnet50}"
RL_TAG="${RL_TAG:-rlv2}"
RL_PREFIX="funnybirds-mcbm-${RL_TAG}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # curated/train
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"
TEMPLATE="$HERE/configs/funnybirds-mcbm.yaml"
source "$HERE/_paths.sh"                                    # gen_config(): honors FB_PKLS override

# point the config generator at the RELABELED pkls (gen_config -> _pick_pkls honors FB_PKLS)
RL_PKLS="$CURATED_DATA/funnybirds_processed_rl"
if [ ! -f "$RL_PKLS/train.pkl" ]; then
  echo "[ERROR] no $RL_PKLS/train.pkl" >&2
  echo "  build it first:" >&2
  echo "  python data/funnybirds/build_funnybirds_cbm_data.py --funnybirds-root \"\$FB\" \\" >&2
  echo "      --labels image_level --out-name funnybirds_processed_rl" >&2
  exit 1
fi
export FB_PKLS="$RL_PKLS"

GEN_DIR="$MCBM/configs/funnybirds"; mkdir -p "$GEN_DIR"
echo "### MCBM-RL sweep  run=$RL_PREFIX  arch=$ARCH  gammas=[$GAMMAS]  seeds=[$SEEDS]"
echo "    pkls=$RL_PKLS  template=$TEMPLATE  gen=$GEN_DIR"
for g in $GAMMAS; do
  gtag="${g//./p}"                                          # 0.1 -> 0p1
  base="${RL_PREFIX}-g${gtag}"                              # prefix (config dir) == funnybirds
  cfg="$GEN_DIR/${base}.yaml"
  gen_config "$TEMPLATE" "$cfg" funnybirds "$ARCH" "$g" || exit 1
  grep -qE "^[[:space:]]*gamma:[[:space:]]*${g}([^0-9]|$)" "$cfg" || { echo "model.gamma sub failed for $g" >&2; exit 1; }
  grep -q "funnybirds_processed_rl" "$cfg" || { echo "[ERROR] $cfg is not pointing at the relabeled pkls" >&2; exit 1; }
  for s in $SEEDS; do
    echo ">>> RL gamma=$g seed=$s  ($base)"
    ( cd "$HERE" && python3 run_mcbm.py "$base" -s "$s" )
  done
done
echo "Done. RL results in $MCBM/results/${RL_PREFIX}-g*/<seed>/."
echo "Next: CONFIG_PREFIX=$RL_PREFIX sbatch train/renderer_swap.slurm"
