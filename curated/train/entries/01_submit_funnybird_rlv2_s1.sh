#!/usr/bin/env bash
# Entry 1: submit exactly FunnyBird RLv2 seed 1.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
JOB=koh_accel_fb_rlv2_s1
ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_v1"
OUT="$ROOT/funnybirds/rlv2/seed1"
BACKUP="$CURATED_DATA/koh_joint_resnet_accelerated_restart_backup/funnybirds/rlv2/seed1"
STANDARD="$ROOT/funnybirds/standard/seed1/SUCCESS.json"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 1: FUNNYBIRD RLV2 S1 ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB" \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
python3 "$REPO/curated/analysis/canonical_manifest.py" verify --manifest "$STANDARD"
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT/SUCCESS.json" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || { echo "ERROR: already queued: $JOB" >&2; exit 2; }

echo "dataset=funnybirds labels=rlv2 seed=1"
echo "framework=Koh Joint CBM backbone=ResNet-50"
echo "concepts=26 species=50 raw-logit linear class head"
echo "loss=normalized task + 0.01 concept"
echo "protocol=accelerated_v1 epochs=100 batch=128 amp=1 workers=8"
echo "lr=0.001 -> 0.02 warmup(5) -> 0.00002 cosine"
echo "dependency=none"
echo "output=$OUT"
echo "steps=audit -> train -> validate -> extract milestones/final test -> convergence -> manifest"
echo "COMMAND: sbatch --job-name=$JOB --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,LABELS=rlv2,KOH_OUTPUT_ROOT=$ROOT,KOH_RESTART_BACKUP_DIR=$BACKUP curated/train/koh_accelerated_funnybird_seed1_job.slurm"

python3 "$REPO/curated/analysis/audit_koh_accelerated.py"
python3 "$REPO/curated/analysis/audit_koh_resnet.py" weights
python3 "$REPO/curated/analysis/audit_koh_resnet.py" boundary \
  --koh-root "$REPO/curated/external/ConceptBottleneck" --num-classes 50
python3 "$REPO/curated/analysis/audit_koh_resnet.py" model \
  --koh-root "$REPO/curated/external/ConceptBottleneck" \
  --num-classes 50 --num-attributes 26

jid=$(sbatch --parsable --job-name="$JOB" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,LABELS=rlv2,KOH_OUTPUT_ROOT=$ROOT,KOH_RESTART_BACKUP_DIR=$BACKUP" \
  "$REPO/curated/train/koh_accelerated_funnybird_seed1_job.slurm")
echo "[ENTRY 1 SUBMITTED] job=$jid dependency=none output=$OUT"
scontrol show job -dd "$jid" | grep -E \
  'JobId=|JobName=|JobState=|Dependency=|Command=|WorkDir=|Environment='
scontrol write batch_script "$jid" - | grep -E \
  'LABELS=|DATASET=|BACKBONE=|KOH_TRAINING_PROTOCOL=|koh_joint_stage'
