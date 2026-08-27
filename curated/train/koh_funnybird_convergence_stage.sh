#!/usr/bin/env bash
# Resume one completed FunnyBird seed-1 model in 25-epoch blocks until stable.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${LABELS:?set LABELS=standard|rlv2}"
case "$LABELS" in standard|rlv2) ;; *) echo "ERROR: invalid labels=$LABELS" >&2; exit 2 ;; esac

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
SOURCE_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_v1"
TARGET_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1"
SOURCE="$SOURCE_ROOT/funnybirds/$LABELS/seed1"
OUT="$TARGET_ROOT/funnybirds/$LABELS/seed1"
SOURCE_BACKUP="$CURATED_DATA/koh_joint_resnet_accelerated_restart_backup/funnybirds/$LABELS/seed1"
TARGET_BACKUP="$CURATED_DATA/koh_joint_resnet_accelerated_converged_restart_backup/funnybirds/$LABELS/seed1"

if [ "$LABELS" = standard ]; then
  python3 "$REPO/curated/analysis/canonical_manifest.py" verify \
    --manifest "$SOURCE/SUCCESS.json"
else
  python3 - "$SOURCE/CONVERGENCE.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "INCOMPLETE":
    raise SystemExit("ERROR: RLv2 source is not the recorded incomplete convergence run")
print("[RLV2 EPOCH-100 SOURCE PASS] completed model requires continuation")
PY
fi

if [ ! -d "$OUT" ]; then
  mkdir -p "$OUT" "$TARGET_BACKUP"
  for epoch in 025 050 075 100; do
    cp -p "$SOURCE/milestone_epoch_${epoch}.pth" "$OUT/"
    cp -p "$SOURCE/milestone_epoch_${epoch}_test.parquet" "$OUT/"
  done
  cp -p "$SOURCE_BACKUP/restart_state.pth" "$OUT/restart_state.pth"
  cp -p "$SOURCE_BACKUP/restart_state.pth" "$TARGET_BACKUP/restart_state.pth"
  echo "[CONTINUATION INITIALIZED] labels=$LABELS source_epoch=100 destination=$OUT"
fi

python3 - "$OUT/restart_state.pth" <<'PY'
import sys, torch
path = sys.argv[1]
try:
    state = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(path, map_location="cpu")
if state.get("format") != "koh_accelerated_epoch_boundary_v1":
    raise SystemExit("ERROR: wrong continuation restart format")
if state.get("next_epoch", -1) < 100:
    raise SystemExit(f"ERROR: continuation restart is only at epoch {state.get('next_epoch')}")
print(f"[CONTINUATION RESTART PASS] next_epoch={state['next_epoch']}")
PY

for target in 125 150 175 200; do
  if [ -s "$OUT/SUCCESS.json" ]; then
    python3 "$REPO/curated/analysis/canonical_manifest.py" verify \
      --manifest "$OUT/SUCCESS.json"
    echo "[CONVERGENCE ALREADY ACCEPTED] labels=$LABELS"
    exit 0
  fi
  completed=$(python3 - "$OUT/restart_state.pth" <<'PY'
import sys, torch
try:
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(sys.argv[1], map_location="cpu")
print(state["next_epoch"])
PY
  )
  if [ "$completed" -ge "$target" ]; then
    continue
  fi
  echo "===== FUNNYBIRD $LABELS CONVERGENCE BLOCK: $completed -> $target ====="
  echo "architecture=Koh Joint ResNet-50; no MCBM or Inception modules"
  echo "optimizer state, AMP scaler, and RNG state resume from epoch $completed"
  echo "learning_rate=0.00002; check=$((target - 25))->$target ordinary-health stability"
  set +e
  DATASET=funnybirds LABELS="$LABELS" SEED=1 BACKBONE=resnet50 \
    KOH_TRAINING_PROTOCOL=accelerated_v1 \
    KOH_ACCELERATED_TARGET_EPOCHS="$target" \
    KOH_OUTPUT_ROOT="$TARGET_ROOT" KOH_RESTART_BACKUP_DIR="$TARGET_BACKUP" \
    bash "$REPO/curated/train/koh_joint_stage.sh"
  status=$?
  set -e
  case "$status" in
    0) echo "[CONVERGENCE ACCEPTED] labels=$LABELS target_epoch=$target"; exit 0 ;;
    3) echo "[CONVERGENCE INCOMPLETE] labels=$LABELS target_epoch=$target; continuing" ;;
    *) echo "ERROR: $LABELS continuation stopped with code $status" >&2; exit "$status" ;;
  esac
done

echo "INCOMPLETE: $LABELS did not stabilize by the predeclared epoch-200 cap" >&2
exit 3
