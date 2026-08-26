#!/usr/bin/env bash
# Train exactly the Koh-paper CUB Joint CBM, adapted only for dataset dimensions.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${DATASET:?set DATASET=funnybirds|cub70|cub}"
: "${LABELS:?set LABELS=standard|rlv2}"
: "${SEED:?set SEED=1|2|3}"
BACKBONE="${BACKBONE:-inception_v3}"
TRAINING_PROTOCOL="${KOH_TRAINING_PROTOCOL:-koh_original}"
EXTRA_MANIFEST_INPUTS=()
EXTRA_MANIFEST_OUTPUTS=()

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
KOH_SOURCE="$CURATED/external/ConceptBottleneck"
EXPECTED_KOH_COMMIT=d6353f270702b92feb5b084a6fd065f891d583f8
test "$(git -C "$KOH_SOURCE" rev-parse HEAD)" = "$EXPECTED_KOH_COMMIT" || {
  echo "ERROR: Koh submodule is not pinned at $EXPECTED_KOH_COMMIT" >&2
  exit 2
}
git -C "$KOH_SOURCE" diff --quiet -- && git -C "$KOH_SOURCE" diff --cached --quiet -- || {
  echo "ERROR: pinned Koh submodule has tracked modifications" >&2
  exit 2
}
if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_resnet_accelerated_v1}"
elif [ "$BACKBONE" = resnet50 ]; then
  ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_resnet_v1}"
else
  ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_v1}"
fi

case "$TRAINING_PROTOCOL" in
  koh_original|accelerated_v1) ;;
  *) echo "ERROR: unsupported training protocol $TRAINING_PROTOCOL" >&2; exit 2 ;;
esac

case "$BACKBONE" in
  inception_v3|resnet50) ;;
  *) echo "ERROR: unsupported Koh backbone $BACKBONE" >&2; exit 2 ;;
esac
if [ "$BACKBONE" = resnet50 ]; then
  [ "$SEED" = 1 ] || {
    echo "ERROR: ResNet Koh seed-one gate rejected seed $SEED" >&2
    exit 2
  }
fi
if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  case "$BACKBONE:$DATASET:$LABELS:$SEED" in
    resnet50:funnybirds:standard:1|resnet50:funnybirds:rlv2:1) ;;
    *)
    echo "ERROR: accelerated_v1 is gated to ResNet FunnyBird seed 1" >&2
    exit 2
    ;;
  esac
fi

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
test -f "$KOH_SOURCE/experiments.py" || { echo "ERROR: missing Koh experiments.py" >&2; exit 2; }
test ! -e "$KOH_SOURCE/src/experiments.py" || {
  echo "ERROR: unexpected shadow entry point $KOH_SOURCE/src/experiments.py" >&2; exit 2;
}
if [ "$BACKBONE" = inception_v3 ]; then
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
else
  python3 "$CURATED/analysis/audit_koh_resnet.py" weights
fi

OUT="$ROOT/$DATASET/$LABELS/seed$SEED"
test ! -e "$OUT/SUCCESS.json" || {
  echo "ERROR: accepted output already exists; refusing to overwrite $OUT" >&2; exit 2;
}
if [ "$BACKBONE" = resnet50 ]; then
  KOH_RESTART_BACKUP_DIR="${KOH_RESTART_BACKUP_DIR:-$CURATED_DATA/koh_joint_resnet_restart_backup/$DATASET/$LABELS/seed$SEED}"
  export KOH_RESTART_BACKUP_DIR
  if [ ! -s "$OUT/restart_state.pth" ] \
     && [ -s "$KOH_RESTART_BACKUP_DIR/restart_state.pth" ]; then
    mkdir -p "$OUT"
    cp -p "$KOH_RESTART_BACKUP_DIR/restart_state.pth" "$OUT/restart_state.pth"
    echo "[RESTART RESTORED FROM BACKUP] $OUT/restart_state.pth"
  fi
fi
if [ -d "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] \
   && [ ! -s "$OUT/restart_state.pth" ]; then
  echo "ERROR: incomplete output exists without a restart state; refusing to delete it: $OUT" >&2
  echo "Preserve or explicitly relocate the provisional artifacts before starting a new run." >&2
  exit 2
fi
mkdir -p "$OUT"

if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  python3 "$CURATED/analysis/audit_koh_accelerated.py" \
    --output "$OUT/TRAINING_PROTOCOL.json"
  EXTRA_MANIFEST_INPUTS+=(--input "$OUT/TRAINING_PROTOCOL.json")
fi

# Use a per-process copy of the pinned Koh source so adapters never dirty the
# paper-citable submodule and concurrent jobs cannot race.
mkdir -p "$CURATED_DATA/koh_joint_runtime"
KOH="$(mktemp -d "$CURATED_DATA/koh_joint_runtime/${DATASET}_${LABELS}_s${SEED}.XXXXXX")"
cleanup_runtime() {
  case "${KOH:-}" in
    "$CURATED_DATA/koh_joint_runtime/"*) rm -rf -- "$KOH" ;;
    "") ;;
    *) echo "ERROR: refusing to remove unexpected runtime path: $KOH" >&2 ;;
  esac
}
trap cleanup_runtime EXIT
git -C "$KOH_SOURCE" archive --format=tar HEAD | tar -xf - -C "$KOH"
if [ "$BACKBONE" = resnet50 ]; then
  mkdir -p "$KOH_RESTART_BACKUP_DIR"
  echo "[RESTART BACKUP] $KOH_RESTART_BACKUP_DIR"
fi
if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  # accelerated_v1 replaces CUB.train.train at import time and owns its atomic,
  # scaler-aware restart state.  The historical patch targets the original
  # train() that is not executed under this protocol, so the Koh runtime must
  # remain byte-identical here.
  echo "[RESTART CONFIG] enabled=1 path=$OUT/restart_state.pth provider=accelerated_v1"
else
  patch_targets="$(sed -n 's|^+++ b/||p' "$CURATED/patches/koh_restartable_training.patch" | sort -u)"
  [ "$patch_targets" = CUB/train.py ] || {
    echo "ERROR: historical restart patch targets unexpected files: $patch_targets" >&2
    exit 2
  }
  # When CURATED_DATA sits inside another git work tree (as on Adroit),
  # git-apply resolves patch paths against that enclosing repository and
  # silently skips CUB/train.py with exit status 0.  A momentary repository
  # rooted at the runtime copy pins the path root to $KOH itself; the marker
  # check below remains the hard gate.
  (cd "$KOH" && git init -q . \
    && git apply --recount -v \
      "$CURATED/patches/koh_restartable_training.patch" \
    && rm -rf .git)
  export KOH_RESTARTABLE=1
  grep -q "koh_epoch_boundary_v1" "$KOH/CUB/train.py" || {
    echo "ERROR: isolated Koh runtime does not contain the restart-state patch" >&2
    exit 2
  }
  echo "[RESTART CONFIG] enabled=1 path=$OUT/restart_state.pth provider=koh_original_patch"
fi
echo "[TRAINING PROTOCOL] $TRAINING_PROTOCOL"

if [ "$BACKBONE" = resnet50 ]; then
  python3 "$CURATED/analysis/audit_koh_resnet.py" boundary \
    --koh-root "$KOH" --num-classes "$N_CLASSES"
  python3 "$CURATED/analysis/audit_koh_resnet.py" model \
    --koh-root "$KOH" --output "$OUT/MODEL_PREFLIGHT.json" \
    --num-classes "$N_CLASSES" --num-attributes "$N_ATTR"
  integrity="$OUT/INPUT_INTEGRITY.json"
  integrity_check="$OUT/INPUT_INTEGRITY.check.json"
  python3 "$CURATED/analysis/audit_koh_resnet.py" data \
    --pkl "$DATA/train.pkl" --pkl "$DATA/val.pkl" --pkl "$DATA/test.pkl" \
    --work-dir "$WORK" --output "$integrity_check"
  if [ -s "$integrity" ]; then
    cmp "$integrity" "$integrity_check" || {
      echo "ERROR: FunnyBird inputs changed since the preceding run segment" >&2
      exit 2
    }
    rm -f "$integrity_check"
  else
    mv "$integrity_check" "$integrity"
  fi
  EXTRA_MANIFEST_INPUTS+=(--input "$integrity" --input "$OUT/MODEL_PREFLIGHT.json")
fi

if [ "$N_CLASSES" = 200 ] && [ "$BACKBONE" = inception_v3 ]; then
  # Historical Full CUB/Inception needs no adapter.
  KOH_ENTRY=(python3 "$KOH/experiments.py")
else
  # run_koh owns the explicit ResNet constructor boundary and, for FB/CUB70,
  # the class-count change. Full CUB passes the unchanged count of 200.
  KOH_ENTRY=(python3 "$CURATED/compat/run_koh.py"
    --curated-num-classes "$N_CLASSES" --curated-koh-root "$KOH")
  if [ "$DATASET" = cub70 ]; then
    # Koh's weighting formula is undefined for the two all-zero CUB70 targets.
    # The adapter sets only those unused positive weights to neutral 1.0.
    KOH_ENTRY+=(--curated-neutral-constant-imbalance)
  fi
fi
if [ "$BACKBONE" = resnet50 ]; then
  KOH_ENTRY+=(--curated-backbone resnet50 --curated-require-seed-one)
fi

# Everything after the entry point is copied verbatim from the official
# CUB/README.md Joint-0.01 command, except dataset path, output path, seed, and
# dimensions required by the current dataset.
CMD=("${KOH_ENTRY[@]}" CUB Joint --seed "$SEED" -ckpt 1
  -log_dir "$OUT" -e 1000 -optimizer sgd -pretrained -use_aux -use_attr
  -weighted_loss multiple -data_dir "$DATA"
  -n_attributes "$N_ATTR" -attr_loss_weight 0.01 -normalize_loss -b 64
  -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end)

if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  CMD=("${KOH_ENTRY[@]}" CUB Joint --seed "$SEED" -ckpt 1
    -log_dir "$OUT" -e 100 -optimizer sgd -pretrained -use_aux -use_attr
    -weighted_loss multiple -data_dir "$DATA"
    -n_attributes "$N_ATTR" -attr_loss_weight 0.01 -normalize_loss -b 128
    -weight_decay 0.0004 -lr 0.02 -scheduler_step 1000 -end2end)
fi

printf 'COMMAND:'; printf ' %q' "${CMD[@]}"; printf '\n'
cd "$WORK"
TRAIN_START_SECONDS=$SECONDS
"${CMD[@]}" &
TRAIN_PID=$!

# Fail closed early instead of discovering after a multi-day timeout that the
# job imported an unpatched trainer.  A FunnyBird epoch takes several minutes;
# 20 minutes is ample for the first atomic epoch-boundary state while still
# bounding wasted GPU time when restartability is broken.
restart_deadline=$((SECONDS + 1200))
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if [ -s "$OUT/restart_state.pth" ]; then
    python3 - "$OUT/restart_state.pth" "$TRAINING_PROTOCOL" <<'PY'
import sys
import torch

path, training_protocol = sys.argv[1:3]
try:
    state = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(path, map_location="cpu")
required = {
    "format", "next_epoch", "best_val_epoch", "best_val_acc",
    "training_complete", "model_state_dict", "optimizer_state_dict",
    "scheduler_state_dict", "python_rng_state", "numpy_rng_state",
    "torch_rng_state", "cuda_rng_state_all",
}
if training_protocol == "accelerated_v1":
    required.update({"training_protocol", "scaler_state_dict"})
missing = sorted(required.difference(state))
expected_format = (
    "koh_accelerated_epoch_boundary_v1"
    if training_protocol == "accelerated_v1"
    else "koh_epoch_boundary_v1"
)
if state.get("format") != expected_format or missing:
    raise SystemExit(
        f"ERROR: invalid restart state format={state.get('format')!r} missing={missing}"
    )
if not isinstance(state.get("next_epoch"), int) or state["next_epoch"] < 1:
    raise SystemExit(f"ERROR: invalid restart next_epoch={state.get('next_epoch')!r}")
print(f"[RESTART STATE PASS] path={path} next_epoch={state['next_epoch']}")
PY
    break
  fi
  if [ "$SECONDS" -ge "$restart_deadline" ]; then
    echo "ERROR: trainer produced no restart_state.pth within 20 minutes" >&2
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    exit 2
  fi
  sleep 15
done

wait "$TRAIN_PID"

if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  for epoch in 025 050 075 100; do
    test -s "$OUT/milestone_epoch_${epoch}.pth" || {
      echo "ERROR: missing accelerated milestone epoch $epoch" >&2
      exit 2
    }
  done
  test -s "$OUT/final_model_${SEED}.pth" || {
    echo "ERROR: missing accelerated final checkpoint" >&2
    exit 2
  }
fi

if [ "$BACKBONE" = resnet50 ]; then
  python3 "$CURATED/analysis/audit_koh_resnet.py" data \
    --pkl "$DATA/train.pkl" --pkl "$DATA/val.pkl" --pkl "$DATA/test.pkl" \
    --work-dir "$WORK" --output "$OUT/INPUT_INTEGRITY_AFTER.json"
  cmp "$OUT/INPUT_INTEGRITY.json" "$OUT/INPUT_INTEGRITY_AFTER.json" || {
    echo "ERROR: FunnyBird inputs changed during training" >&2
    exit 2
  }
fi

if [ -n "${KOH_BENCHMARK_EPOCHS:-}" ]; then
  python3 - "$OUT/restart_state.pth" "$KOH_BENCHMARK_EPOCHS" \
    "$((SECONDS - TRAIN_START_SECONDS))" <<'PY'
import sys
import torch

path, requested, elapsed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    state = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(path, map_location="cpu")
completed = state["next_epoch"]
if state.get("training_complete"):
    raise SystemExit("ERROR: benchmark unexpectedly marked training complete")
if completed < requested:
    raise SystemExit(
        f"ERROR: benchmark requested {requested} epochs but state has {completed}"
    )
seconds_per_epoch = elapsed / completed
print(
    f"[KOH BENCHMARK COMPLETE] epochs={completed} elapsed_seconds={elapsed} "
    f"seconds_per_epoch={seconds_per_epoch:.3f} "
    f"six_hour_760_epoch_threshold=26.053 pass={seconds_per_epoch <= 26.053}"
)
PY
  echo "[BENCHMARK STOP] restart state preserved; unset KOH_BENCHMARK_EPOCHS to resume"
  exit 75
fi

if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  CKPT="$OUT/final_model_${SEED}.pth"
else
  CKPT="$OUT/best_model_${SEED}.pth"
fi
python3 "$CURATED/analysis/validate_koh_joint.py" \
  --checkpoint "$CKPT" --koh-root "$KOH" --dataset "$DATASET" \
  --labels "$LABELS" --seed "$SEED" --num-classes "$N_CLASSES" \
  --num-attributes "$N_ATTR" --backbone "$BACKBONE" \
  --training-protocol "$TRAINING_PROTOCOL" \
  --manifest "$OUT/CHECKPOINT.json"
python3 "$CURATED/analysis/export_koh_eval.py" --koh-root "$KOH" \
  --checkpoint "$CKPT" --kind joint --data-pkl "$DATA/test.pkl" \
  --work-dir "$WORK" --n-attributes "$N_ATTR" "${NAME_ARGS[@]}" \
  --out "$OUT/final_test.parquet"
if [ "$TRAINING_PROTOCOL" = accelerated_v1 ]; then
  for epoch in 025 050 075 100; do
    parquet="$OUT/milestone_epoch_${epoch}_test.parquet"
    python3 "$CURATED/analysis/export_koh_eval.py" --koh-root "$KOH" \
      --checkpoint "$OUT/milestone_epoch_${epoch}.pth" --kind joint \
      --data-pkl "$DATA/test.pkl" --work-dir "$WORK" \
      --n-attributes "$N_ATTR" "${NAME_ARGS[@]}" --out "$parquet"
    EXTRA_MANIFEST_OUTPUTS+=(
      --output "$OUT/milestone_epoch_${epoch}.pth" --output "$parquet"
    )
  done
  python3 "$CURATED/analysis/audit_koh_accelerated_convergence.py" \
    --epoch-25 "$OUT/milestone_epoch_025_test.parquet" \
    --epoch-50 "$OUT/milestone_epoch_050_test.parquet" \
    --epoch-75 "$OUT/milestone_epoch_075_test.parquet" \
    --epoch-100 "$OUT/milestone_epoch_100_test.parquet" \
    --output "$OUT/CONVERGENCE.json" --require-stable
  EXTRA_MANIFEST_OUTPUTS+=(--output "$OUT/CONVERGENCE.json")
fi
cd "$REPO"
python3 "$CURATED/analysis/canonical_manifest.py" write --repo "$REPO" \
  --stage "koh_joint_${DATASET}_${LABELS}_s${SEED}" \
  --manifest "$OUT/SUCCESS.json" \
  --command "koh_joint_stage.sh $DATASET $LABELS $SEED" \
  --input "$DATA/train.pkl" --input "$DATA/val.pkl" --input "$DATA/test.pkl" \
  "${EXTRA_MANIFEST_INPUTS[@]}" \
  --output "$CKPT" --output "$OUT/CHECKPOINT.json" \
  --output "$OUT/final_test.parquet" "${EXTRA_MANIFEST_OUTPUTS[@]}" \
  --meta "framework=koh_joint" \
  --meta "backbone=$BACKBONE" \
  --meta "training_protocol=$TRAINING_PROTOCOL" \
  --meta "dataset=$DATASET" --meta "labels=$LABELS" --meta "seed=$SEED"
rm -f "$OUT/restart_state.pth" "$OUT/restart_state.pth.tmp"
