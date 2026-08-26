#!/usr/bin/env bash
# Entry 2: submit exactly CUB70 standard seed 1.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
JOB=koh_resnet_cub70_s1
ROOT="$CURATED_DATA/koh_joint_resnet_v1"
OUT="$ROOT/cub70/standard/seed1"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 2: CUB70 STANDARD S1 ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB" \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT/SUCCESS.json" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || { echo "ERROR: already queued: $JOB" >&2; exit 2; }

echo "dataset=cub70 labels=standard seed=1"
echo "framework=Koh Joint CBM backbone=ResNet-50"
echo "concepts=112 species=70 raw-logit linear class head"
echo "loss=normalized task + 0.01 concept"
echo "protocol=koh_original dependency=none time_limit=24h"
echo "output=$OUT"
echo "steps=audit -> train -> validate -> extract final test -> manifest"
echo "COMMAND: sbatch --time=1-00:00:00 --job-name=$JOB --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=cub70,LABELS=standard,SEED=1,BACKBONE=resnet50,KOH_TRAINING_PROTOCOL=koh_original,KOH_OUTPUT_ROOT=$ROOT curated/train/koh_joint_job.slurm"

python3 "$REPO/curated/analysis/audit_koh_resnet.py" weights
python3 "$REPO/curated/analysis/audit_koh_resnet.py" boundary \
  --koh-root "$REPO/curated/external/ConceptBottleneck" --num-classes 70
python3 "$REPO/curated/analysis/audit_koh_resnet.py" model \
  --koh-root "$REPO/curated/external/ConceptBottleneck" \
  --num-classes 70 --num-attributes 112

jid=$(sbatch --parsable --time=1-00:00:00 --job-name="$JOB" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=cub70,LABELS=standard,SEED=1,BACKBONE=resnet50,KOH_TRAINING_PROTOCOL=koh_original,KOH_OUTPUT_ROOT=$ROOT" \
  "$REPO/curated/train/koh_joint_job.slurm")
echo "[ENTRY 2 SUBMITTED] job=$jid dependency=none output=$OUT"
scontrol show job -dd "$jid" | grep -E \
  'JobId=|JobName=|JobState=|Dependency=|Command=|WorkDir=|TimeLimit=|Environment=' || \
  echo "WARNING: job submitted but accepted-payload display was unavailable"
scontrol write batch_script "$jid" - | grep -E \
  'koh_joint_stage|DATASET=|LABELS=|SEED=|BACKBONE=|KOH_TRAINING_PROTOCOL=' || \
  echo "WARNING: job submitted but batch-script display was unavailable"
