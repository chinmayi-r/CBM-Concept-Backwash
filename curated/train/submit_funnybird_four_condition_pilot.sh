#!/usr/bin/env bash
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUN_ID="pilot_tail_wing_25_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$CURATED_DATA/funnybird_four_condition_v1/$RUN_ID"
CHECKPOINT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/final_model_1.pth"

echo "===== CURRENT CLUSTER STATE ====="
squeue -u "$USER"
echo "===== RECENT FOUR-CONDITION PILOTS ====="
sacct -u "$USER" --starttime now-2days -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End |
  grep -E 'JobID|fb_4cell_pilot' || true

echo "===== PROPOSED PILOT ====="
echo "NAME: FunnyBird four-condition tail-wing visual-calibration pilot"
echo "WHY: test whether exact named-part pixels or the four other named parts supply the target concept score"
echo "ACTION: render and run frozen inference only; no training"
echo "FRAMEWORK: accepted Koh Joint CBM with ResNet-50; Standard seed 1"
echo "DATA: FunnyBird held-out test annotations; tail and wing; 25 balanced images each"
echo "QUANTITIES: z11, z01, z10, z00 and five paired within-image contrasts"
echo "INTERVENTION: paste identical target pixels into both target-present cells"
echo "BOUNDARY: base body, pose, camera, lighting, and background remain in every cell"
echo "INPUT: $CURATED_DATA/FunnyBirds"
echo "CHECKPOINT: $CHECKPOINT"
echo "OUTPUT: $OUT"
echo "DEPENDENCY: none"
echo "TIME LIMIT: 02:00:00"
echo "RESULT STATUS: visual-calibration pilot only; no scientific acceptance before gallery review"
echo "COMMAND: sbatch --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,RUN_ID=$RUN_ID,OUT=$OUT,PARTS_KEY=tail-wing,MAX_IMAGES_PER_PART=25,CHECKPOINT=$CHECKPOINT curated/train/funnybird_four_condition_job.slurm"

job_id=$(sbatch --parsable \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,RUN_ID=$RUN_ID,OUT=$OUT,PARTS_KEY=tail-wing,MAX_IMAGES_PER_PART=25,CHECKPOINT=$CHECKPOINT" \
  curated/train/funnybird_four_condition_job.slurm)
echo "SUBMITTED: job=$job_id output=$OUT"
scontrol show job -dd "$job_id" | grep -E \
  'JobId=|JobName=|JobState=|Reason=|Command=|WorkDir=|StdOut=|TimeLimit=|Environment='
