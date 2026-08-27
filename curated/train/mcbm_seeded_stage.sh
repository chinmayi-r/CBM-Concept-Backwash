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
STABILITY_PROTOCOL="${MCBM_STABILITY_PROTOCOL:-recorded_v1}"

case "$DATASET:$LABELS" in
  funnybirds:standard|funnybirds:rlv2|cub70:standard|cub:standard) ;;
  *) echo "ERROR: unsupported cell $DATASET:$LABELS" >&2; exit 2 ;;
esac
case "$SEED" in 1|2|3) ;; *) echo "ERROR: invalid seed $SEED" >&2; exit 2 ;; esac
case "$GAMMA" in 0|0.1|0.3|1|3|5) ;; *) echo "ERROR: invalid gamma $GAMMA" >&2; exit 2 ;; esac
REQUIRED_INPUTS=(
  "$INPUT/selection/train.pkl"
  "$INPUT/selection/test.pkl"
  "$INPUT/final/test.pkl"
)
SCHEMA_MANIFEST_ARGS=()
if [ "$DATASET" = cub70 ] || [ "$DATASET" = cub ]; then
  REQUIRED_INPUTS+=("$INPUT/selection/selection_indices.json")
  SCHEMA_MANIFEST_ARGS+=(--input "$INPUT/selection/selection_indices.json")
fi
for file in "${REQUIRED_INPUTS[@]}"; do
  test -s "$file" || { echo "ERROR: missing $file" >&2; exit 2; }
done

tag="${GAMMA//./p}"
case "$STABILITY_PROTOCOL" in
  recorded_v1)
    RUN_PREFIX="${DATASET}-seeded-v1-mcbm-${LABELS}"
    output_root="$CURATED_DATA/mcbm_seeded_v1"
    BASE_LR="${BASE_LR:-0.01}"
    MCBM_TRAINING_PRECISION="${MCBM_TRAINING_PRECISION:-amp}"
    ;;
  cub70_stabilized_high_gamma_v1)
    [ "$DATASET:$LABELS:$SEED" = cub70:standard:1 ] || {
      echo "ERROR: stabilized protocol is only for CUB70 standard seed 1" >&2; exit 2;
    }
    case "$GAMMA" in 1|3|5) ;; *) echo "ERROR: stabilized bridge requires gamma 1, 3, or 5" >&2; exit 2 ;; esac
    [ "${BASE_LR:-0.003}" = 0.003 ] || { echo "ERROR: stabilized base LR must be 0.003" >&2; exit 2; }
    [ "${MCBM_TRAINING_PRECISION:-fp32}" = fp32 ] || { echo "ERROR: stabilized precision must be fp32" >&2; exit 2; }
    RUN_PREFIX="cub70-stabilized-high-gamma-v1-mcbm-standard"
    output_root="$CURATED_DATA/mcbm_stabilized_high_gamma_v1"
    BASE_LR=0.003
    MCBM_TRAINING_PRECISION=fp32
    ;;
  *) echo "ERROR: unsupported MCBM_STABILITY_PROTOCOL=$STABILITY_PROTOCOL" >&2; exit 2 ;;
esac
base="${RUN_PREFIX}-g${tag}"
epoch=250; [ "$DATASET" = funnybirds ] && epoch=100
out="$output_root/$DATASET/$LABELS/g${tag}/seed${SEED}"
test ! -e "$out/SUCCESS.json" || {
  echo "ERROR: accepted output already exists; refusing overwrite: $out" >&2; exit 2;
}
mkdir -p "$out"

export RUN_PREFIX BASE_LR MCBM_TRAINING_PRECISION GAMMAS="$GAMMA" SEEDS="$SEED"
if [ "$DATASET" = funnybirds ]; then
  export FB_PKLS="$INPUT/selection"
else
  export CUB_PKLS="$INPUT/selection"
fi
cd "$CURATED"
echo "[MCBM TRAINING PROTOCOL] protocol=$STABILITY_PROTOCOL precision=$MCBM_TRAINING_PRECISION base_lr=$BASE_LR"
bash train/mcbm_gamma_sweep.sh "$DATASET"

config="$MCBM/configs/$DATASET/$base.yaml"
checkpoint="$MCBM/results/$base/$SEED/models/epoch_${epoch}.pt"
python3 analysis/audit_mcbm_artifact.py --repo "$REPO" --config "$config" \
  --checkpoint "$checkpoint" --dataset "$DATASET" --labels "$LABELS" \
  --gamma "$GAMMA" --seed "$SEED" --expected-base-lr "$BASE_LR" \
  --training-precision "$MCBM_TRAINING_PRECISION" --out "$out/CHECKPOINT.json"
python3 analysis/export_mcbm_eval.py --config "$base" --seed "$SEED" \
  --epoch "$epoch" --final-test "$INPUT/final" --out "$out/final_test.parquet"
python3 analysis/canonical_manifest.py write --repo "$REPO" \
  --stage "mcbm_${DATASET}_${LABELS}_g${tag}_s${SEED}" \
  --manifest "$out/SUCCESS.json" \
  --command "mcbm_seeded_stage.sh $DATASET $LABELS $GAMMA $SEED" \
  --input "$config" --input "$INPUT/selection/train.pkl" \
  --input "$INPUT/selection/test.pkl" --input "$INPUT/final/test.pkl" \
  --input "$CURATED/patches/minimal_cbm.patch" \
  "${SCHEMA_MANIFEST_ARGS[@]}" \
  --output "$checkpoint" --output "$out/CHECKPOINT.json" \
  --output "$out/final_test.parquet" --meta "framework=minimal_cbm" \
  --meta "dataset=$DATASET" --meta "labels=$LABELS" \
  --meta "gamma=$GAMMA" --meta "seed=$SEED" \
  --meta "stability_protocol=$STABILITY_PROTOCOL" \
  --meta "training_precision=$MCBM_TRAINING_PRECISION" \
  --meta "base_lr=$BASE_LR"
