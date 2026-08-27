#!/usr/bin/env bash
# Entry 10: CUB70 stabilized high-gamma bridge, gamma 5, seed 1.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
JOB=m_cub70_stable_g5_s1
OUT="$CURATED_DATA/mcbm_stabilized_high_gamma_v1/cub70/standard/g5/seed1"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 10: CUB70 MCBM STABILIZED BRIDGE GAMMA 5 S1 ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB"   --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || { echo "ERROR: already queued: $JOB" >&2; exit 2; }

echo "goal=internally comparable CUB70 high-gamma bridge"
echo "framework=official minimal_cbm MCBM backbone=ResNet-50 dataset=CUB70 seed=1 gamma=5"
echo "dimensions=112 concepts, 70 species, hidden_dim=1024"
echo "frozen changes=training precision FP32 and base LR 0.003"
echo "unchanged=batch 64, SGD momentum 0.9, weight decay 0.00004, 250 epochs"
echo "dependency=none output=$OUT"
echo "steps=source audit -> train -> finite checkpoint audit -> final-test export -> manifest"
echo "resource=one full GPU (FP32 batch 64 may exceed the historical 20-GB MIG slice)"
echo "COMMAND: sbatch --job-name=$JOB --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=cub70,LABELS=standard,GAMMA=5,SEED=1,MCBM_STABILITY_PROTOCOL=cub70_stabilized_high_gamma_v1,MCBM_TRAINING_PRECISION=fp32,BASE_LR=0.003 curated/train/mcbm_stabilized_job.slurm"

bash "$REPO/curated/train/verify_canonical_sources.sh"
jid=$(sbatch --parsable --job-name="$JOB"   --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=cub70,LABELS=standard,GAMMA=5,SEED=1,MCBM_STABILITY_PROTOCOL=cub70_stabilized_high_gamma_v1,MCBM_TRAINING_PRECISION=fp32,BASE_LR=0.003"   "$REPO/curated/train/mcbm_stabilized_job.slurm")
echo "[ENTRY 10 SUBMITTED] job=$jid dependency=none output=$OUT"
