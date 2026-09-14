#!/usr/bin/env bash
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUN_ID="preflight_all_parts_18_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$CURATED_DATA/funnybird_four_condition_v1/$RUN_ID"
CHECKPOINT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/final_model_1.pth"

git -C "$REPO" diff --quiet -- \
  curated/analysis/funnybird_four_condition.py \
  curated/analysis/funnybird_four_condition_core.py \
  curated/analysis/test_funnybird_four_condition.py \
  curated/train/funnybird_four_condition_job.slurm \
  curated/train/submit_funnybird_four_condition_allparts_preflight.sh || {
    echo "ERROR: the four-condition source differs from the checked-out commit" >&2
    exit 2
  }

echo "===== CURRENT CLUSTER STATE ====="
squeue -u "$USER" -o "%.18i %.30j %.2t %.12M %.12l %R"
echo "===== RECENT MATCHING JOBS ====="
sacct -u "$USER" --starttime now-7days -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End |
  grep -E 'JobID|fb_4cell_all' || true

echo "===== PROPOSED PREFLIGHT 1: FOUR CONDITIONS FOR ALL PARTS ====="
echo "WHY: compare named-pixel evidence with other-part context for every FunnyBird part"
echo "ACTION: render and run frozen inference only; no training"
echo "FRAMEWORK: accepted Koh Joint CBM with ResNet-50; Standard seed 1"
echo "DATA: FunnyBird held-out test annotations; all five parts; 18 balanced images each"
echo "QUANTITIES: z11, z01, z10, z00 and five paired within-image contrasts"
echo "INTERVENTION: paste identical target pixels into both target-present cells"
echo "BOUNDARY: only the four other named part meshes are removed; body/pose/background remain"
echo "INPUT: $CURATED_DATA/FunnyBirds"
echo "CHECKPOINT: $CHECKPOINT"
echo "OUTPUT: $OUT"
echo "DEPENDENCY: none"
echo "TIME LIMIT: 02:00:00"
echo "RESULT STATUS: visual-calibration preflight; no scientific result until every part gallery is reviewed"
echo "COMMAND: sbatch --job-name=fb_4cell_all --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,RUN_ID=$RUN_ID,OUT=$OUT,PARTS_KEY=tail-wing-beak-foot-eye,MAX_IMAGES_PER_PART=18,CHECKPOINT=$CHECKPOINT curated/train/funnybird_four_condition_job.slurm"

job_id=$(sbatch --parsable --job-name=fb_4cell_all \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,RUN_ID=$RUN_ID,OUT=$OUT,PARTS_KEY=tail-wing-beak-foot-eye,MAX_IMAGES_PER_PART=18,CHECKPOINT=$CHECKPOINT" \
  "$REPO/curated/train/funnybird_four_condition_job.slurm")
echo "SUBMITTED: job=$job_id output=$OUT"
scontrol show job -dd "$job_id" | grep -E \
  'JobId=|JobName=|JobState=|Reason=|Command=|WorkDir=|StdOut=|TimeLimit=|Environment='
