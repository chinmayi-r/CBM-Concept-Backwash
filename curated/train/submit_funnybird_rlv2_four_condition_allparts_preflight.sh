#!/usr/bin/env bash
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
: "${STANDARD_FOUR_CONDITION_ANALYSIS:?export STANDARD_FOUR_CONDITION_ANALYSIS to the completed Standard analysis directory}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUN_ID="rlv2_matched_all_parts_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$CURATED_DATA/funnybird_four_condition_v1/$RUN_ID"
MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/rlv2/seed1"
CHECKPOINT="$MODEL_ROOT/final_model_1.pth"
MODEL_MANIFEST="$MODEL_ROOT/SUCCESS.json"
SELECTION_CSV="$STANDARD_FOUR_CONDITION_ANALYSIS/rows.csv"

test -s "$SELECTION_CSV" || { echo "ERROR: missing Standard eligibility table: $SELECTION_CSV" >&2; exit 2; }
git -C "$REPO" diff --quiet -- \
  curated/analysis/funnybird_four_condition.py \
  curated/analysis/funnybird_four_condition_core.py \
  curated/analysis/test_funnybird_four_condition.py \
  curated/train/funnybird_four_condition_job.slurm \
  curated/train/submit_funnybird_rlv2_four_condition_allparts_preflight.sh || {
    echo "ERROR: the RLv2 four-condition source differs from the checked-out commit" >&2
    exit 2
  }

echo "===== CURRENT CLUSTER STATE ====="
squeue -u "$USER" -o "%.18i %.30j %.2t %.12M %.12l %R"
echo "===== RECENT MATCHING JOBS ====="
sacct -u "$USER" --starttime now-7days -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End |
  grep -E 'JobID|fb_4cell_rl' || true

echo "===== PROPOSED RLV2 PREFLIGHT 1: MATCHED FOUR CONDITIONS ====="
echo "QUESTION: after correcting invisible positive labels, how much target score still comes from other parts?"
echo "PREDICTION: label conflict as the cause predicts lower context-without-target evidence, especially for tail"
echo "ACTION: reuse the 85 Standard image-only eligible rows and run frozen RLv2 inference; no training"
echo "SELECTION: eligibility uses only renderer pixels, never Standard model scores"
echo "MODEL: accepted Koh Joint ResNet-50 RLv2 seed 1"
echo "CHECKPOINT: $CHECKPOINT"
echo "STANDARD ELIGIBILITY: $SELECTION_CSV"
echo "OUTPUT: $OUT"
echo "DEPENDENCY: none"
echo "TIME LIMIT: 02:00:00"
echo "COMMAND: sbatch --job-name=fb_4cell_rl --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,RUN_ID=$RUN_ID,OUT=$OUT,PARTS_KEY=tail-wing-beak-foot-eye,MAX_IMAGES_PER_PART=18,REGIME_LABEL=RLv2,SELECTION_CSV=$SELECTION_CSV,CHECKPOINT=$CHECKPOINT,MODEL_MANIFEST=$MODEL_MANIFEST curated/train/funnybird_four_condition_job.slurm"

job_id=$(sbatch --parsable --job-name=fb_4cell_rl \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,RUN_ID=$RUN_ID,OUT=$OUT,PARTS_KEY=tail-wing-beak-foot-eye,MAX_IMAGES_PER_PART=18,REGIME_LABEL=RLv2,SELECTION_CSV=$SELECTION_CSV,CHECKPOINT=$CHECKPOINT,MODEL_MANIFEST=$MODEL_MANIFEST" \
  "$REPO/curated/train/funnybird_four_condition_job.slurm")
echo "SUBMITTED: job=$job_id output=$OUT"
scontrol show job -dd "$job_id" | grep -E \
  'JobId=|JobName=|JobState=|Reason=|Command=|WorkDir=|StdOut=|TimeLimit=|Environment='
