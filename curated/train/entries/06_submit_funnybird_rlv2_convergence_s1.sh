#!/usr/bin/env bash
# Entry 6: continue FunnyBird RLv2 seed 1 from epoch 100 until stable.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
JOB=koh_fb_rlv2_converge_s1
OUT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/rlv2/seed1"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 6: FUNNYBIRD RLV2 S1 CONVERGENCE CONTINUATION ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB" \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || { echo "ERROR: already queued: $JOB" >&2; exit 2; }

echo "goal=resume completed RLv2 seed 1; test stability every 25 epochs; cap at 200"
echo "framework=Koh Joint CBM backbone=ResNet-50 labels=rlv2 seed=1"
echo "training=resumes epoch-100 optimizer/scaler/RNG; LR remains 0.00002"
echo "dependency=none output=$OUT"
echo "steps=resume -> milestone -> final-test export -> convergence audit -> manifest"
echo "COMMAND: sbatch --job-name=$JOB --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,LABELS=rlv2 curated/train/koh_funnybird_convergence_job.slurm"
jid=$(sbatch --parsable --job-name="$JOB" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,LABELS=rlv2" \
  "$REPO/curated/train/koh_funnybird_convergence_job.slurm")
echo "[ENTRY 6 SUBMITTED] job=$jid dependency=none output=$OUT"
