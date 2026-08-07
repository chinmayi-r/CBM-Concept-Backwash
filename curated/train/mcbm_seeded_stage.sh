#!/usr/bin/env bash
# Train and validate one exact dataset/label/gamma/seed MCBM cell.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${DATASET:?set DATASET=funnybirds|cub70|cub}"
: "${LABELS:?set LABELS=standard|rlv2}"
: "${GAMMA:?set one GAMMA}"
: "${SEED:?set one SEED}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
MCBM="$CURATED/external/minimal_cbm"
INPUT="$CURATED_DATA/mcbm_seeded_v1_inputs/$DATASET/$LABELS"

case "$DATASET:$LABELS" in
  funnybirds:standard|funnybirds:rlv2|cub70:standard|cub:standard) ;;
  *) echo "ERROR: unsupported cell $DATASET:$LABELS" >&2; exit 2 ;;
esac
case "$SEED" in 1|2|3) ;; *) echo "ERROR: invalid seed $SEED" >&2; exit 2 ;; esac
case "$GAMMA" in 0|0.1|0.3|1|3|5) ;; *) echo "ERROR: invalid gamma $GAMMA" >&2; exit 2 ;; esac
for file in "$INPUT/selection/train.pkl" "$INPUT/selection/test.pkl" "$INPUT/final/test.pkl"; do
  test -s "$file" || { echo "ERROR: missing $file" >&2; exit 2; }
done

tag="${GAMMA//./p}"
RUN_PREFIX="${DATASET}-seeded-v1-mcbm-${LABELS}"
base="${RUN_PREFIX}-g${tag}"
epoch=250; [ "$DATASET" = funnybirds ] && epoch=100
out="$CURATED_DATA/mcbm_seeded_v1/$DATASET/$LABELS/g${tag}/seed${SEED}"
test ! -e "$out/SUCCESS.json" || {
  echo "ERROR: accepted output already exists; refusing overwrite: $out" >&2; exit 2;
}
mkdir -p "$out"

export RUN_PREFIX GAMMAS="$GAMMA" SEEDS="$SEED"
if [ "$DATASET" = funnybirds ]; then
  export FB_PKLS="$INPUT/selection"
else
  export CUB_PKLS="$INPUT/selection"
fi
cd "$CURATED"
bash train/mcbm_gamma_sweep.sh "$DATASET"

config="$MCBM/configs/$DATASET/$base.yaml"
checkpoint="$MCBM/results/$base/$SEED/models/epoch_${epoch}.pt"
python3 analysis/audit_mcbm_artifact.py --repo "$REPO" --config "$config" \
  --checkpoint "$checkpoint" --dataset "$DATASET" --labels "$LABELS" \
  --gamma "$GAMMA" --seed "$SEED" --out "$out/CHECKPOINT.json"
python3 analysis/export_mcbm_eval.py --config "$base" --seed "$SEED" \
  --epoch "$epoch" --final-test "$INPUT/final" --out "$out/final_test.parquet"
python3 analysis/canonical_manifest.py write --repo "$REPO" \
  --stage "mcbm_${DATASET}_${LABELS}_g${tag}_s${SEED}" \
  --manifest "$out/SUCCESS.json" \
  --command "mcbm_seeded_stage.sh $DATASET $LABELS $GAMMA $SEED" \
  --input "$config" --input "$INPUT/selection/train.pkl" \
  --input "$INPUT/selection/test.pkl" --input "$INPUT/final/test.pkl" \
  --output "$checkpoint" --output "$out/CHECKPOINT.json" \
  --output "$out/final_test.parquet" --meta "framework=minimal_cbm" \
  --meta "dataset=$DATASET" --meta "labels=$LABELS" \
  --meta "gamma=$GAMMA" --meta "seed=$SEED"
