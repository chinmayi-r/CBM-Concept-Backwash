#!/usr/bin/env bash
# Fixed-render comparison for the accepted accelerated FunnyBird seed-1 pair.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
ROOT="${KOH_FUNNYBIRD_MODEL_ROOT:-$CURATED_DATA/koh_joint_resnet_accelerated_v1/funnybirds}"
OUT="${KOH_FUNNYBIRD_SWAP_OUT:-$CURATED_DATA/swap_koh_joint_resnet_accelerated_v1_seed1}"
CACHE="${KOH_RENDER_CACHE:-$CURATED_DATA/swap_fixed_v2_attempt2/render_cache}"

for labels in standard rlv2; do
  success="$ROOT/$labels/seed1/SUCCESS.json"
  checkpoint="$ROOT/$labels/seed1/final_model_1.pth"
  test -s "$success" || { echo "ERROR: missing accepted $labels seed-1 manifest" >&2; exit 2; }
  test -s "$checkpoint" || { echo "ERROR: missing $labels final checkpoint" >&2; exit 2; }
  python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$success"
done

mkdir -p "$OUT"
KOH_CHECKPOINT="$ROOT/standard/seed1/final_model_1.pth" \
  KOH_KIND=joint KOH_NAME=funnybirds-cbm CONFIG_PREFIX=funnybirds-cbm \
  GAMMAS=0 SEEDS=1 SWAP_OUT="$OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
  bash "$CURATED/train/renderer_swap.slurm"
KOH_CHECKPOINT="$ROOT/rlv2/seed1/final_model_1.pth" \
  KOH_KIND=joint KOH_NAME=funnybirds-cbm-rlv2matched \
  CONFIG_PREFIX=funnybirds-cbm-rlv2matched GAMMAS=0 SEEDS=1 \
  SWAP_OUT="$OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
  bash "$CURATED/train/renderer_swap.slurm"

python3 "$CURATED/analysis/validate_fixed_swaps.py" --out "$OUT"
python3 "$CURATED/analysis/compare_fixed_rl.py" --out "$OUT" --rl-tag rlv2matched
python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
  --stage koh_funnybird_seed1_fixed_swaps --manifest "$OUT/SUCCESS.json" \
  --command "koh_funnybird_seed1_swaps_stage.sh" \
  --input "$ROOT/standard/seed1/SUCCESS.json" \
  --input "$ROOT/rlv2/seed1/SUCCESS.json" \
  --output "$OUT/funnybirds-cbm-s1.csv" \
  --output "$OUT/funnybirds-cbm-rlv2matched-s1.csv" \
  --output "$OUT/fixed_rl_comparison_rlv2matched.csv" \
  --meta framework=koh_joint --meta dataset=funnybirds \
  --meta seeds=1 --meta labels=standard,rlv2
