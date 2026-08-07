#!/usr/bin/env bash
# Train exactly the Koh-paper CUB Joint CBM, adapted only for dataset dimensions.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${DATASET:?set DATASET=funnybirds|cub70|cub}"
: "${LABELS:?set LABELS=standard|rlv2}"
: "${SEED:?set SEED=1|2|3}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
KOH="$CURATED/external/ConceptBottleneck"
ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_v1}"

case "$DATASET:$LABELS" in
  funnybirds:standard)
    N_CLASSES=50; N_ATTR=26
    DATA="$CURATED_DATA/koh_joint_inputs/funnybirds/standard"
    NAME_ARGS=(--names "$CURATED_DATA/koh_joint_inputs/funnybird_concept_names.json")
    ;;
  funnybirds:rlv2)
    N_CLASSES=50; N_ATTR=26
    DATA="$CURATED_DATA/koh_joint_inputs/funnybirds/rlv2"
    NAME_ARGS=(--names "$CURATED_DATA/koh_joint_inputs/funnybird_concept_names.json")
    ;;
  cub70:standard)
    N_CLASSES=70; N_ATTR=112
    DATA="$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original"
    NAME_ARGS=(--selection-indices "$DATA/selection_indices.json"
      --attributes "$CURATED_DATA/CUB_processed/attributes.txt")
    ;;
  cub:standard)
    N_CLASSES=200; N_ATTR=112
    DATA="$CURATED_DATA/CUB_processed/class_attr_data_10"
    NAME_ARGS=(--selection-indices "$DATA/selection_indices.json"
      --attributes "$CURATED_DATA/CUB_processed/attributes.txt")
    ;;
  *) echo "ERROR: unsupported stage $DATASET:$LABELS" >&2; exit 2 ;;
esac

for split in train val test; do
  test -s "$DATA/$split.pkl" || {
    echo "ERROR: missing $DATA/$split.pkl" >&2; exit 2;
  }
done
WORK="$CURATED_DATA/koh_joint_inputs/work/$DATASET"
test -L "$WORK/CUB_200_2011" || {
  echo "ERROR: missing image view $WORK/CUB_200_2011" >&2; exit 2;
}
test -f "$KOH/experiments.py" || { echo "ERROR: missing Koh experiments.py" >&2; exit 2; }
test ! -e "$KOH/src/experiments.py" || {
  echo "ERROR: unexpected shadow entry point $KOH/src/experiments.py" >&2; exit 2;
}
weights="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/inception_v3_google-1a9a5a14.pth"
test -s "$weights" || {
  echo "ERROR: official Inception weights are not cached: $weights" >&2
  echo "Run train/prepare_koh_pretrained.sh on the login node." >&2
  exit 2
}
prefix=$(sha256sum "$weights" | awk '{print substr($1,1,8)}')
test "$prefix" = 1a9a5a14 || {
  echo "ERROR: Inception weight hash prefix is $prefix, expected 1a9a5a14" >&2
  exit 2
}

OUT="$ROOT/$DATASET/$LABELS/seed$SEED"
test ! -e "$OUT/SUCCESS.json" || {
  echo "ERROR: accepted output already exists; refusing to overwrite $OUT" >&2; exit 2;
}
mkdir -p "$OUT"

if [ "$N_CLASSES" = 200 ]; then
  # Full CUB needs no adapter: invoke the pinned repository directly.
  KOH_ENTRY=(python3 "$KOH/experiments.py")
else
  # Koh hard-codes 200 classes; only this constant changes for FB/CUB70.
  KOH_ENTRY=(python3 "$CURATED/compat/run_koh.py"
    --curated-num-classes "$N_CLASSES")
  if [ "$DATASET" = cub70 ]; then
    # Koh's weighting formula is undefined for the two all-zero CUB70 targets.
    # The adapter sets only those unused positive weights to neutral 1.0.
    KOH_ENTRY+=(--curated-neutral-constant-imbalance)
  fi
fi

# Everything after the entry point is copied verbatim from the official
# CUB/README.md Joint-0.01 command, except dataset path, output path, seed, and
# dimensions required by the current dataset.
CMD=("${KOH_ENTRY[@]}" CUB Joint --seed "$SEED" -ckpt 1
  -log_dir "$OUT" -e 1000 -optimizer sgd -pretrained -use_aux -use_attr
  -weighted_loss multiple -data_dir "$DATA"
  -n_attributes "$N_ATTR" -attr_loss_weight 0.01 -normalize_loss -b 64
  -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end)

printf 'COMMAND:'; printf ' %q' "${CMD[@]}"; printf '\n'
cd "$WORK"
"${CMD[@]}"

CKPT="$OUT/best_model_${SEED}.pth"
python3 "$CURATED/analysis/validate_koh_joint.py" \
  --checkpoint "$CKPT" --koh-root "$KOH" --dataset "$DATASET" \
  --labels "$LABELS" --seed "$SEED" --num-classes "$N_CLASSES" \
  --num-attributes "$N_ATTR" --manifest "$OUT/CHECKPOINT.json"
python3 "$CURATED/analysis/export_koh_eval.py" --koh-root "$KOH" \
  --checkpoint "$CKPT" --kind joint --data-pkl "$DATA/test.pkl" \
  --work-dir "$WORK" --n-attributes "$N_ATTR" "${NAME_ARGS[@]}" \
  --out "$OUT/final_test.parquet"
cd "$REPO"
python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
  --stage "koh_joint_${DATASET}_${LABELS}_s${SEED}" \
  --manifest "$OUT/SUCCESS.json" \
  --command "koh_joint_stage.sh $DATASET $LABELS $SEED" \
  --input "$DATA/train.pkl" --input "$DATA/val.pkl" --input "$DATA/test.pkl" \
  --output "$CKPT" --output "$OUT/CHECKPOINT.json" \
  --output "$OUT/final_test.parquet" --meta "framework=koh_joint" \
  --meta "dataset=$DATASET" --meta "labels=$LABELS" --meta "seed=$SEED"
