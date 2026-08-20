#!/usr/bin/env bash
# Review-first submission of exactly one FunnyBird ResNet/Koh seed-1 condition.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
LABELS="${1:?usage: submit_koh_resnet_funnybird_seed1.sh standard|rlv2}"
case "$LABELS" in standard|rlv2) ;; *) echo "ERROR: invalid labels $LABELS" >&2; exit 2 ;; esac
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
ROOT="$CURATED_DATA/koh_joint_resnet_v1"
OUT="$ROOT/funnybirds/$LABELS/seed1"
BACKUP="$CURATED_DATA/koh_joint_resnet_restart_backup/funnybirds/$LABELS/seed1"

echo "===== FRESH QUEUE ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
echo "===== RELEVANT ACCOUNTING ====="
sacct -u "$USER" --starttime now-7days -X \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
echo "===== EXACT PAYLOAD ====="
printf 'repo=%s\ncommit=%s\nlabels=%s\nseed=1\nbackbone=resnet50\noutput=%s\nbackup=%s\n' \
  "$REPO" "$(git -C "$REPO" rev-parse HEAD)" "$LABELS" "$OUT" "$BACKUP"

[ ! -s "$OUT/SUCCESS.json" ] || {
  echo "ERROR: completed seed-1 output already exists: $OUT/SUCCESS.json" >&2
  exit 2
}
bash "$REPO/curated/train/preflight_koh_resnet.sh"
python3 "$REPO/curated/analysis/audit_koh_resnet.py" weights
bash -n "$REPO/curated/train/koh_resnet_funnybird_seed1_job.slurm"

if [ "${SUBMIT_APPROVED:-}" != YES ]; then
  echo "[DRY RUN ONLY] Nothing submitted. Set SUBMIT_APPROVED=YES after review."
  exit 0
fi

job="koh_resnet_fb_${LABELS}_s1"
if squeue -h -u "$USER" -n "$job" | grep -q .; then
  echo "ERROR: job already queued: $job" >&2
  exit 2
fi
jid=$(sbatch --parsable --job-name="$job" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,LABELS=$LABELS,KOH_OUTPUT_ROOT=$ROOT,KOH_RESTART_BACKUP_DIR=$BACKUP" \
  "$REPO/curated/train/koh_resnet_funnybird_seed1_job.slurm")
echo "[SUBMITTED] job=$jid labels=$LABELS seed=1"
