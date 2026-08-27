#!/usr/bin/env bash
# Entry 11: resume the preserved Full-CUB seed-1 model at epoch 439 and stop at 600.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
JOB=koh_cub_decay600_s1
SOURCE="$CURATED_DATA/manual_pause_backups/full_cub_seed1_job3357529"
ROOT="$CURATED_DATA/koh_joint_resnet_decay_continuation_v1"
OUT="$ROOT/cub/standard/seed1"
BACKUP="$CURATED_DATA/koh_joint_resnet_decay_continuation_restart_backup/cub/standard/seed1"

test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || {
  echo "ERROR: tracked changes" >&2; exit 2;
}

echo "===== ENTRY 11: FULL-CUB RESNET CBM DECAY CONTINUATION S1 ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB" \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || {
  echo "ERROR: already queued: $JOB" >&2; exit 2;
}
test -s "$SOURCE/restart_state.pth" -a -s "$SOURCE/best_model_1.pth" || {
  echo "ERROR: preserved epoch-439 source is missing" >&2; exit 2;
}

python3 - "$SOURCE/restart_state.pth" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert state.get("format") == "koh_epoch_boundary_v1", state.get("format")
assert state.get("next_epoch") == 439, state.get("next_epoch")
assert state.get("training_complete") is False
lrs = {float(group["lr"]) for group in state["optimizer_state_dict"]["param_groups"]}
assert lrs == {0.001}, lrs
print("[FULL-CUB CONTINUATION SOURCE PASS] next_epoch=439 optimizer_lr=0.001")
PY

if [ ! -d "$OUT" ]; then
  mkdir -p "$OUT"
  cp -p "$SOURCE/restart_state.pth" "$OUT/restart_state.pth"
  cp -p "$SOURCE/best_model_1.pth" "$OUT/best_model_1.pth"
fi
test -s "$OUT/restart_state.pth" -a -s "$OUT/best_model_1.pth" || {
  echo "ERROR: continuation output lacks its restart or inherited best model" >&2; exit 2;
}

restart_sha=$(sha256sum "$SOURCE/restart_state.pth" | awk '{print $1}')
best_sha=$(sha256sum "$SOURCE/best_model_1.pth" | awk '{print $1}')
python3 - "$OUT/CONTINUATION_PROTOCOL.json" "$restart_sha" "$best_sha" <<'PY'
import json
import sys
from pathlib import Path

path, restart_sha, best_sha = sys.argv[1:]
record = {
    "status": "PASS",
    "framework": "koh_joint",
    "backbone": "resnet50",
    "dataset": "full_cub",
    "seed": 1,
    "source_job": 3357529,
    "source_next_epoch": 439,
    "source_restart_sha256": restart_sha,
    "source_best_model_sha256": best_sha,
    "schedule": "cosine",
    "start_epoch": 439,
    "end_epoch_exclusive": 600,
    "start_lr": 0.001,
    "end_lr": 0.00002,
    "architecture_or_loss_change": False,
}
Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

echo "dataset=full_cub labels=standard seed=1"
echo "framework=Koh Joint CBM backbone=ResNet-50 concepts=112 species=200"
echo "loss=unchanged normalized task + 0.01 concept; raw-logit linear class head"
echo "protocol=full_cub_decay_continuation_v1"
echo "schedule=resume epoch 439; cosine LR 0.001 -> 0.00002; stop after epoch 600"
echo "dependency=none output=$OUT"
echo "COMMAND: sbatch --time=12:00:00 --job-name=$JOB --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=cub,LABELS=standard,SEED=1,BACKBONE=resnet50,KOH_TRAINING_PROTOCOL=full_cub_decay_continuation_v1,KOH_OUTPUT_ROOT=$ROOT,KOH_RESTART_BACKUP_DIR=$BACKUP curated/train/koh_joint_job.slurm"

jid=$(sbatch --parsable --time=12:00:00 --job-name="$JOB" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=cub,LABELS=standard,SEED=1,BACKBONE=resnet50,KOH_TRAINING_PROTOCOL=full_cub_decay_continuation_v1,KOH_OUTPUT_ROOT=$ROOT,KOH_RESTART_BACKUP_DIR=$BACKUP" \
  "$REPO/curated/train/koh_joint_job.slurm")
echo "[ENTRY 11 SUBMITTED] job=$jid dependency=none output=$OUT"
scontrol show job -dd "$jid" | grep -E \
  'JobId=|JobName=|JobState=|Dependency=|Command=|WorkDir=|TimeLimit='
