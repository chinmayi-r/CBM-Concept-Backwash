#!/usr/bin/env bash
# Evaluate all official-Koh FunnyBird standard/RLv2 checkpoints on one shared,
# validated fixed-render population, then create the paired comparison.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
ROOT="$CURATED_DATA/koh_joint_v1/funnybirds"
OUT="$CURATED_DATA/swap_koh_joint_v1"
CACHE="${KOH_RENDER_CACHE:-$CURATED_DATA/swap_fixed_v2_attempt2/render_cache}"
MANIFEST="$OUT/SUCCESS.json"

if [ -s "$MANIFEST" ]; then
  python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$MANIFEST"
  exit 0
fi

inputs=()
for seed in 1 2 3; do
  for labels in standard rlv2; do
    success="$ROOT/$labels/seed$seed/SUCCESS.json"
    checkpoint="$ROOT/$labels/seed$seed/best_model_${seed}.pth"
    test -s "$success" || { echo "ERROR: training did not complete: $success" >&2; exit 2; }
    test -s "$checkpoint" || { echo "ERROR: checkpoint missing: $checkpoint" >&2; exit 2; }
    inputs+=("$success" "$checkpoint")
  done
done

mkdir -p "$OUT"
for seed in 1 2 3; do
  for labels in standard rlv2; do
    name="funnybirds-cbm"
    [ "$labels" = rlv2 ] && name="funnybirds-cbm-rlv2matched"
    checkpoint="$ROOT/$labels/seed$seed/best_model_${seed}.pth"
    echo "===== FIXED SWAP $name seed=$seed ====="
    KOH_CHECKPOINT="$checkpoint" KOH_KIND=joint KOH_NAME="$name" \
      CONFIG_PREFIX="$name" GAMMAS="0" SEEDS="$seed" \
      SWAP_OUT="$OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
      bash "$CURATED/train/renderer_swap.slurm"
  done
done

python3 "$CURATED/analysis/validate_fixed_swaps.py" --out "$OUT"
python3 "$CURATED/analysis/compare_fixed_rl.py" \
  --out "$OUT" --rl-tag rlv2matched

manifest_args=()
for path in "${inputs[@]}"; do manifest_args+=(--input "$path"); done
for seed in 1 2 3; do
  manifest_args+=(--output "$OUT/funnybirds-cbm-s${seed}.csv")
  manifest_args+=(--output "$OUT/funnybirds-cbm-rlv2matched-s${seed}.csv")
done
manifest_args+=(--output "$OUT/fixed_rl_comparison_rlv2matched.csv")

python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
  --stage koh_funnybird_fixed_swaps --manifest "$MANIFEST" \
  --command "koh_funnybird_swaps_stage.sh" "${manifest_args[@]}" \
  --meta framework=koh_joint --meta dataset=funnybirds \
  --meta seeds=1,2,3 --meta labels=standard,rlv2

