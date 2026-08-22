#!/usr/bin/env bash
# Review-first submission of exactly one accelerated FunnyBird standard seed 1.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_v1"
OUT="$ROOT/funnybirds/standard/seed1"
BACKUP="$CURATED_DATA/koh_joint_resnet_accelerated_restart_backup/funnybirds/standard/seed1"
JOB=koh_accel_fb_standard_s1

echo "===== FRESH QUEUE ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
echo "===== RELEVANT ACCOUNTING ====="
sacct -u "$USER" --starttime now-7days -X \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
echo "===== EXACT PAYLOAD ====="
printf '%s\n' \
  "repo=$REPO" \
  "commit=$(git -C "$REPO" rev-parse HEAD)" \
  "dataset=funnybirds" \
  "labels=standard" \
  "seed=1" \
  "backbone=resnet50" \
  "architecture=koh_joint_raw_logits_linear_class_head" \
  "training_protocol=accelerated_v1" \
  "epochs=100 batch=128 amp=1 workers=8" \
  "lr=0.001->0.02->0.00002 warmup=5 scheduler=cosine" \
  "output=$OUT" \
  "backup=$BACKUP"

[ ! -s "$OUT/SUCCESS.json" ] || {
  echo "ERROR: completed accelerated output already exists: $OUT/SUCCESS.json" >&2
  exit 2
}
bash "$REPO/curated/train/preflight_koh_accelerated.sh"

if [ "${SUBMIT_APPROVED:-}" != YES ]; then
  echo "[DRY RUN ONLY] Nothing submitted. Set SUBMIT_APPROVED=YES after review."
  exit 0
fi
if squeue -h -u "$USER" -n "$JOB" | grep -q .; then
  echo "ERROR: job already queued: $JOB" >&2
  exit 2
fi

jid=$(sbatch --parsable --job-name="$JOB" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,KOH_OUTPUT_ROOT=$ROOT,KOH_RESTART_BACKUP_DIR=$BACKUP" \
  "$REPO/curated/train/koh_accelerated_funnybird_seed1_job.slurm")
echo "[SUBMITTED] job=$jid labels=standard seed=1 protocol=accelerated_v1"
