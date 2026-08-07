#!/usr/bin/env bash
# One deterministic payload used by every canonical Slurm job.
set -euo pipefail

: "${REPO:?set REPO}"
: "${CANONICAL_ROOT:?set CANONICAL_ROOT}"
: "${STAGE:?set STAGE}"
: "${DATASET:?set DATASET}"
: "${LABELS:?set LABELS}"
: "${SEED:?set SEED}"

CURATED="$REPO/curated"
KOH="$CURATED/external/ConceptBottleneck"
MCBM="$CURATED/external/minimal_cbm"
MANIFEST_DIR="$CANONICAL_ROOT/manifests"
mkdir -p "$MANIFEST_DIR"
[ "$(git -C "$REPO" rev-parse HEAD)" = "${CANONICAL_REPO_SHA:?set CANONICAL_REPO_SHA}" ] || {
  echo "ERROR: repository changed after canonical submission" >&2; exit 2;
}
bash "$CURATED/train/verify_canonical_sources.sh"

case "$DATASET" in
  funnybirds)
    N_CLASSES=50; N_ATTR=26; KOH_WORK="$CANONICAL_ROOT/data/koh_work/funnybirds"
    case "$LABELS" in
      standard) DATA_KEY=funnybirds_standard ;;
      rlv2) DATA_KEY=funnybirds_rlv2 ;;
      *) echo "ERROR: invalid FunnyBird labels: $LABELS" >&2; exit 2 ;;
    esac ;;
  cub)
    N_CLASSES=200; N_ATTR=112; KOH_WORK="$CANONICAL_ROOT/data/koh_work/cub"
    [ "$LABELS" = standard ] || { echo "ERROR: CUB RLv2 is not defined" >&2; exit 2; }
    DATA_KEY=cub ;;
  cub70)
    N_CLASSES=70; N_ATTR=112; KOH_WORK="$CANONICAL_ROOT/data/koh_work/cub70"
    [ "$LABELS" = standard ] || { echo "ERROR: CUB70 RLv2 is not defined" >&2; exit 2; }
    DATA_KEY=cub70 ;;
  *) echo "ERROR: unknown dataset $DATASET" >&2; exit 2 ;;
esac

DATA="$CANONICAL_ROOT/data/$DATA_KEY/koh"
for split in train val test; do
  test -s "$DATA/$split.pkl" || { echo "ERROR: missing $DATA/$split.pkl" >&2; exit 2; }
done

KOH_BASE="$CANONICAL_ROOT/koh/$DATASET/$LABELS/seed$SEED"
KOH_RUN=(python3 "$CURATED/compat/run_koh.py"
  --curated-num-classes "$N_CLASSES" --curated-num-attributes "$N_ATTR")
COMMON=(CUB --seed "$SEED" -use_attr -data_dir "$DATA" -n_attributes "$N_ATTR")

write_manifest() {
  local name="$1" output="$2" command="$3"
  python3 "$CURATED/analysis/canonical_manifest.py" write \
    --repo "$REPO" --stage "$name" \
    --manifest "$MANIFEST_DIR/${name}.json" --command "$command" \
    --input "$DATA/train.pkl" --input "$DATA/val.pkl" --input "$DATA/test.pkl" \
    --output "$output" --meta "dataset=$DATASET" --meta "labels=$LABELS" \
    --meta "seed=$SEED" --meta "framework=koh"
}

cd "$KOH_WORK"
case "$STAGE" in
  koh_concept)
    OUT="$KOH_BASE/concept"; mkdir -p "$OUT"
    CMD=("${KOH_RUN[@]}" "${COMMON[@]}" Concept_XtoC)
    # Reorder because the official parser requires dataset and experiment first.
    CMD=(python3 "$CURATED/compat/run_koh.py" --curated-num-classes "$N_CLASSES"
      --curated-num-attributes "$N_ATTR" CUB Concept_XtoC --seed "$SEED"
      -log_dir "$OUT" -e 1000 -optimizer sgd -pretrained -use_aux -use_attr
      -weighted_loss multiple -data_dir "$DATA" -n_attributes "$N_ATTR"
      -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01
      -scheduler_step 100 -bottleneck)
    "${CMD[@]}"
    CKPT="$OUT/best_model_${SEED}.pth"
    write_manifest "koh_concept_${DATASET}_${LABELS}_s${SEED}" "$CKPT" "${CMD[*]}" ;;

  koh_independent)
    OUT="$KOH_BASE/independent"; mkdir -p "$OUT"
    CMD=(python3 "$CURATED/compat/run_koh.py" --curated-num-classes "$N_CLASSES"
      --curated-num-attributes "$N_ATTR" CUB Independent_CtoY --seed "$SEED"
      -log_dir "$OUT" -e 500 -optimizer sgd -use_attr -data_dir "$DATA"
      -n_attributes "$N_ATTR" -no_img -b 64 -weight_decay 0.00005
      -lr 0.001 -scheduler_step 100)
    "${CMD[@]}"
    CKPT="$OUT/best_model_${SEED}.pth"
    write_manifest "koh_independent_${DATASET}_${LABELS}_s${SEED}" "$CKPT" "${CMD[*]}" ;;

  koh_extract)
    CONCEPT="$KOH_BASE/concept/best_model_${SEED}.pth"
    python3 "$CURATED/analysis/canonical_manifest.py" verify \
      --manifest "$MANIFEST_DIR/koh_concept_${DATASET}_${LABELS}_s${SEED}.json"
    OUT="$KOH_BASE/predicted_concepts"; mkdir -p "$OUT"
    CMD=(python3 "$CURATED/compat/extract_koh_concepts.py" --koh-root "$KOH"
      --model "$CONCEPT" --data-dir "$DATA" --out-dir "$OUT"
      --work-dir "$KOH_WORK" --seed "$SEED")
    "${CMD[@]}"
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "koh_extract_${DATASET}_${LABELS}_s${SEED}" \
      --manifest "$MANIFEST_DIR/koh_extract_${DATASET}_${LABELS}_s${SEED}.json" \
      --command "${CMD[*]}" --input "$CONCEPT" \
      --output "$OUT/train.pkl" --output "$OUT/val.pkl" --output "$OUT/test.pkl" \
      --meta "dataset=$DATASET" --meta "labels=$LABELS" --meta "seed=$SEED" \
      --meta "framework=koh" ;;

  koh_sequential)
    PRED="$KOH_BASE/predicted_concepts"
    python3 "$CURATED/analysis/canonical_manifest.py" verify \
      --manifest "$MANIFEST_DIR/koh_extract_${DATASET}_${LABELS}_s${SEED}.json"
    OUT="$KOH_BASE/sequential"; mkdir -p "$OUT"
    CMD=(python3 "$CURATED/compat/run_koh.py" --curated-num-classes "$N_CLASSES"
      --curated-num-attributes "$N_ATTR" CUB Sequential_CtoY --seed "$SEED"
      -log_dir "$OUT" -e 500 -optimizer sgd -use_attr -data_dir "$PRED"
      -n_attributes "$N_ATTR" -no_img -b 64 -weight_decay 0.00005
      -lr 0.001 -scheduler_step 100)
    "${CMD[@]}"
    CKPT="$OUT/best_model_${SEED}.pth"
    write_manifest "koh_sequential_${DATASET}_${LABELS}_s${SEED}" "$CKPT" "${CMD[*]}" ;;

  koh_joint|koh_joint_sigmoid)
    SUFFIX=joint; SIGMOID=()
    if [ "$STAGE" = koh_joint_sigmoid ]; then SUFFIX=joint_sigmoid; SIGMOID=(-use_sigmoid); fi
    OUT="$KOH_BASE/$SUFFIX"; mkdir -p "$OUT"
    CMD=(python3 "$CURATED/compat/run_koh.py" --curated-num-classes "$N_CLASSES"
      --curated-num-attributes "$N_ATTR" CUB Joint --seed "$SEED"
      -log_dir "$OUT" -e 1000 -optimizer sgd -pretrained -use_aux -use_attr
      -weighted_loss multiple -data_dir "$DATA" -n_attributes "$N_ATTR"
      -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004
      -lr 0.001 -scheduler_step 100 -end2end "${SIGMOID[@]}")
    "${CMD[@]}"
    CKPT="$OUT/best_model_${SEED}.pth"
    write_manifest "${STAGE}_${DATASET}_${LABELS}_s${SEED}" "$CKPT" "${CMD[*]}" ;;

  mcbm)
    : "${GAMMA:?set GAMMA for MCBM}"
    case "$LABELS" in standard) MKEY="$DATA_KEY" ;; rlv2) MKEY=funnybirds_rlv2 ;; esac
    export FB_PKLS="$CANONICAL_ROOT/data/$MKEY/mcbm_selection"
    export CUB_PKLS="$CANONICAL_ROOT/data/$MKEY/mcbm_selection"
    if [ "$DATASET" != funnybirds ]; then
      : "${CUB_ROOT:?export CUB_ROOT for CUB training}"
      export CUB_IMGS="$CUB_ROOT/images"
      export CUB_ATTR="$CANONICAL_ROOT/data/raw_build/cub/CUB_processed"
    fi
    export GAMMAS="$GAMMA" SEEDS="$SEED" BASE_LR=0.01
    export RUN_PREFIX="${DATASET}-canonical-v1-mcbm-${LABELS}"
    cd "$CURATED"
    bash train/mcbm_gamma_sweep.sh "$DATASET"
    GTAG="${GAMMA//./p}"
    BASE="${RUN_PREFIX}-g${GTAG}"
    EPOCH=250; [ "$DATASET" = funnybirds ] && EPOCH=100
    CKPT="$MCBM/results/$BASE/$SEED/models/epoch_${EPOCH}.pt"
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "mcbm_${DATASET}_${LABELS}_g${GTAG}_s${SEED}" \
      --manifest "$MANIFEST_DIR/mcbm_${DATASET}_${LABELS}_g${GTAG}_s${SEED}.json" \
      --command "GAMMAS=$GAMMA SEEDS=$SEED RUN_PREFIX=$RUN_PREFIX mcbm_gamma_sweep.sh $DATASET" \
      --input "$CANONICAL_ROOT/data/$MKEY/mcbm_selection/train.pkl" \
      --input "$CANONICAL_ROOT/data/$MKEY/mcbm_selection/test.pkl" --output "$CKPT" \
      --meta "dataset=$DATASET" --meta "labels=$LABELS" --meta "seed=$SEED" \
      --meta "gamma=$GAMMA" --meta "framework=minimal_cbm" ;;

  minimal_cbm_cbm)
    case "$LABELS" in standard) MKEY="$DATA_KEY" ;; rlv2) MKEY=funnybirds_rlv2 ;; esac
    export FB_PKLS="$CANONICAL_ROOT/data/$MKEY/mcbm_selection"
    export CUB_PKLS="$CANONICAL_ROOT/data/$MKEY/mcbm_selection"
    if [ "$DATASET" != funnybirds ]; then
      export CUB_IMGS="$CUB_ROOT/images"
      export CUB_ATTR="$CANONICAL_ROOT/data/raw_build/cub/CUB_processed"
    fi
    export MODELS=cbm SEEDS="$SEED" ARCH=resnet50
    export RUN_PREFIX="${DATASET}-canonical-v1-${LABELS}"
    cd "$CURATED"; bash train/run_baselines.sh "$DATASET"
    BASE="${RUN_PREFIX}-cbm"
    EPOCH=250; [ "$DATASET" = funnybirds ] && EPOCH=100
    CKPT="$MCBM/results/$BASE/$SEED/models/epoch_${EPOCH}.pt"
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "minimal_cbm_cbm_${DATASET}_${LABELS}_s${SEED}" \
      --manifest "$MANIFEST_DIR/minimal_cbm_cbm_${DATASET}_${LABELS}_s${SEED}.json" \
      --command "MODELS=cbm RUN_PREFIX=$RUN_PREFIX run_baselines.sh $DATASET" \
      --input "$CANONICAL_ROOT/data/$MKEY/mcbm_selection/train.pkl" \
      --input "$CANONICAL_ROOT/data/$MKEY/mcbm_selection/test.pkl" --output "$CKPT" \
      --meta "dataset=$DATASET" --meta "labels=$LABELS" --meta "seed=$SEED" \
      --meta "framework=minimal_cbm" --meta "role=mcbm_internal_cbm_control" ;;

  eval_koh)
    : "${VARIANT:?set VARIANT for eval_koh}"
    case "$VARIANT" in
      independent|sequential)
        KIND=two_stage
        MODEL="$KOH_BASE/concept/best_model_${SEED}.pth"
        CLASS_MODEL="$KOH_BASE/$VARIANT/best_model_${SEED}.pth"
        python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$MANIFEST_DIR/koh_concept_${DATASET}_${LABELS}_s${SEED}.json"
        python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$MANIFEST_DIR/koh_${VARIANT}_${DATASET}_${LABELS}_s${SEED}.json" ;;
      joint|joint_sigmoid)
        KIND=joint
        MODEL="$KOH_BASE/$VARIANT/best_model_${SEED}.pth"
        CLASS_MODEL=""
        python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$MANIFEST_DIR/koh_${VARIANT}_${DATASET}_${LABELS}_s${SEED}.json" ;;
      *) echo "ERROR: unknown Koh variant $VARIANT" >&2; exit 2 ;;
    esac
    OUT="$CANONICAL_ROOT/eval/koh/$DATASET/$LABELS/seed$SEED/${VARIANT}.parquet"
    CMD=(python3 "$CURATED/analysis/export_koh_eval.py" --koh-root "$KOH" --work-dir "$KOH_WORK"
      --data-pkl "$DATA/test.pkl" --checkpoint "$MODEL" --kind "$KIND"
      --n-attributes "$N_ATTR" --out "$OUT")
    if [ "$DATASET" = funnybirds ]; then
      CMD+=(--names "$CANONICAL_ROOT/data/funnybirds_concept_names.json")
    else
      CMD+=(--selection-indices "$CANONICAL_ROOT/data/$DATASET/selection_indices.json"
        --attributes "$CANONICAL_ROOT/data/$DATASET/attributes.txt")
    fi
    [ -z "$CLASS_MODEL" ] || CMD+=(--class-checkpoint "$CLASS_MODEL")
    "${CMD[@]}"
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "eval_koh_${DATASET}_${LABELS}_${VARIANT}_s${SEED}" \
      --manifest "$MANIFEST_DIR/eval_koh_${DATASET}_${LABELS}_${VARIANT}_s${SEED}.json" \
      --command "${CMD[*]}" --input "$MODEL" --input "$DATA/test.pkl" --output "$OUT" \
      --meta "dataset=$DATASET" --meta "labels=$LABELS" --meta "seed=$SEED" \
      --meta "variant=$VARIANT" --meta "framework=koh" ;;

  eval_mcbm)
    : "${GAMMA:?set GAMMA for eval_mcbm}"
    GTAG="${GAMMA//./p}"
    BASE="${DATASET}-canonical-v1-mcbm-${LABELS}-g${GTAG}"
    EPOCH=250; [ "$DATASET" = funnybirds ] && EPOCH=100
    python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$MANIFEST_DIR/mcbm_${DATASET}_${LABELS}_g${GTAG}_s${SEED}.json"
    OUT="$CANONICAL_ROOT/eval/minimal_cbm/$DATASET/$LABELS/seed$SEED/g${GTAG}.parquet"
    CMD=(python3 "$CURATED/analysis/export_mcbm_eval.py" --config "$BASE" --seed "$SEED"
      --epoch "$EPOCH" --final-test "$CANONICAL_ROOT/data/$DATA_KEY/final_test" --out "$OUT")
    "${CMD[@]}"
    CKPT="$MCBM/results/$BASE/$SEED/models/epoch_${EPOCH}.pt"
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "eval_mcbm_${DATASET}_${LABELS}_g${GTAG}_s${SEED}" \
      --manifest "$MANIFEST_DIR/eval_mcbm_${DATASET}_${LABELS}_g${GTAG}_s${SEED}.json" \
      --command "${CMD[*]}" --input "$CKPT" --input "$CANONICAL_ROOT/data/$DATA_KEY/final_test/test.pkl" \
      --output "$OUT" --meta "dataset=$DATASET" --meta "labels=$LABELS" \
      --meta "seed=$SEED" --meta "gamma=$GAMMA" --meta "framework=minimal_cbm" ;;

  eval_minimal_cbm_cbm)
    BASE="${DATASET}-canonical-v1-${LABELS}-cbm"
    EPOCH=250; [ "$DATASET" = funnybirds ] && EPOCH=100
    python3 "$CURATED/analysis/canonical_manifest.py" verify --manifest "$MANIFEST_DIR/minimal_cbm_cbm_${DATASET}_${LABELS}_s${SEED}.json"
    OUT="$CANONICAL_ROOT/eval/minimal_cbm/$DATASET/$LABELS/seed$SEED/cbm_control.parquet"
    CMD=(python3 "$CURATED/analysis/export_mcbm_eval.py" --config "$BASE" --seed "$SEED"
      --epoch "$EPOCH" --final-test "$CANONICAL_ROOT/data/$DATA_KEY/final_test" --out "$OUT")
    "${CMD[@]}"
    CKPT="$MCBM/results/$BASE/$SEED/models/epoch_${EPOCH}.pt"
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "eval_minimal_cbm_cbm_${DATASET}_${LABELS}_s${SEED}" \
      --manifest "$MANIFEST_DIR/eval_minimal_cbm_cbm_${DATASET}_${LABELS}_s${SEED}.json" \
      --command "${CMD[*]}" --input "$CKPT" --input "$CANONICAL_ROOT/data/$DATA_KEY/final_test/test.pkl" \
      --output "$OUT" --meta "dataset=$DATASET" --meta "labels=$LABELS" \
      --meta "seed=$SEED" --meta "framework=minimal_cbm" --meta "role=mcbm_internal_cbm_control" ;;

  swap_all)
    [ "$DATASET" = funnybirds ] || { echo "ERROR: controlled swaps are FunnyBird-only" >&2; exit 2; }
    SWAP_OUT="$CANONICAL_ROOT/swap/funnybirds/seed$SEED"
    CACHE="$CANONICAL_ROOT/swap/render_cache_seed$SEED"
    mkdir -p "$SWAP_OUT" "$CACHE"
    PREFIX="funnybirds-canonical-v1-mcbm-${LABELS}"
    for koh_stage in koh_concept koh_independent koh_extract koh_sequential koh_joint koh_joint_sigmoid; do
      python3 "$CURATED/analysis/canonical_manifest.py" verify \
        --manifest "$MANIFEST_DIR/${koh_stage}_funnybirds_${LABELS}_s${SEED}.json"
    done
    for gamma in 0 0.1 0.3 1 3 5; do
      tag="${gamma//./p}"
      python3 "$CURATED/analysis/canonical_manifest.py" verify \
        --manifest "$MANIFEST_DIR/mcbm_funnybirds_${LABELS}_g${tag}_s${SEED}.json"
    done
    python3 "$CURATED/analysis/canonical_manifest.py" verify \
      --manifest "$MANIFEST_DIR/minimal_cbm_cbm_funnybirds_${LABELS}_s${SEED}.json"
    CONFIG_PREFIX="$PREFIX" GAMMAS="0 0.1 0.3 1 3 5" SEEDS="$SEED" EPOCH=100 \
      SWAP_OUT="$SWAP_OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
      FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-$CURATED_DATA/FunnyBirds}" \
      bash "$CURATED/train/renderer_swap.slurm"

    CBM_PREFIX="funnybirds-canonical-v1-${LABELS}-cbm"
    CONFIG_PREFIX="$CBM_PREFIX" GAMMAS=0 SEEDS="$SEED" EPOCH=100 \
      SWAP_OUT="$SWAP_OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
      FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-$CURATED_DATA/FunnyBirds}" \
      bash "$CURATED/train/renderer_swap.slurm"

    for variant in independent sequential joint joint_sigmoid; do
      if [ "$variant" = independent ] || [ "$variant" = sequential ]; then
        KIND=two_stage
        XCKPT="$KOH_BASE/concept/best_model_${SEED}.pth"
        YCKPT="$KOH_BASE/$variant/best_model_${SEED}.pth"
        CLASS_ENV=(KOH_CLASS_CHECKPOINT="$YCKPT")
      else
        KIND=joint
        XCKPT="$KOH_BASE/$variant/best_model_${SEED}.pth"
        CLASS_ENV=()
      fi
      NAME="koh-${variant}-${LABELS}-s${SEED}"
      env KOH_CHECKPOINT="$XCKPT" KOH_KIND="$KIND" KOH_NAME="$NAME" \
        "${CLASS_ENV[@]}" CONFIG_PREFIX=unused GAMMAS=0 SEEDS="$SEED" \
        SWAP_OUT="$SWAP_OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
        FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-$CURATED_DATA/FunnyBirds}" \
        bash "$CURATED/train/renderer_swap.slurm"
    done

    OUTPUT_ARGS=(--output "$SWAP_OUT/${CBM_PREFIX}-s${SEED}.csv")
    for gamma in 0 0.1 0.3 1 3 5; do
      tag="${gamma//./p}"
      OUTPUT_ARGS+=(--output "$SWAP_OUT/${PREFIX}-g${tag}-s${SEED}.csv")
    done
    for variant in independent sequential joint joint_sigmoid; do
      OUTPUT_ARGS+=(--output "$SWAP_OUT/koh-${variant}-${LABELS}-s${SEED}.csv")
    done
    python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
      --stage "swap_all_funnybirds_${LABELS}_s${SEED}" \
      --manifest "$MANIFEST_DIR/swap_all_funnybirds_${LABELS}_s${SEED}.json" \
      --command "canonical fixed-render replay for all $LABELS seed $SEED models" \
      "${OUTPUT_ARGS[@]}" --meta "dataset=funnybirds" --meta "labels=$LABELS" \
      --meta "seed=$SEED" --meta "framework=koh+minimal_cbm" ;;

  finalize)
    python3 "$CURATED/analysis/verify_canonical_completion.py" --root "$CANONICAL_ROOT" ;;

  *) echo "ERROR: unknown canonical stage: $STAGE" >&2; exit 2 ;;
esac
