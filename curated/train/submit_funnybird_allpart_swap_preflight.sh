#!/usr/bin/env bash
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUN_ID="allpart_swap_18_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$CURATED_DATA/funnybird_allpart_preflight_v1/$RUN_ID"
CHECKPOINT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/final_model_1.pth"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"

git -C "$REPO" diff --quiet -- \
  curated/analysis/funnybird_allpart_swap_preflight.py \
  curated/analysis/funnybird_allpart_swap_preflight_core.py \
  curated/analysis/test_funnybird_allpart_swap_preflight.py \
  curated/train/funnybird_allpart_swap_preflight_job.slurm \
  curated/train/submit_funnybird_allpart_swap_preflight.sh || {
    echo "ERROR: the all-part preflight source differs from the checked-out commit" >&2
    exit 2
  }

echo "===== CURRENT CLUSTER STATE ====="
squeue -u "$USER" -o "%.18i %.30j %.2t %.12M %.12l %R"
echo "===== RECENT MATCHING JOBS ====="
sacct -u "$USER" --starttime now-7days -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End |
  grep -E 'JobID|fb_xpart_pre' || true

echo "===== PROPOSED PREFLIGHT 2: ALL-TO-ALL ACCEPTED SWAPS ====="
echo "WHY: check all 25 input-part -> output-block paths before choosing a mechanism or correction"
echo "ACTION: replay 18 accepted swaps per part through the frozen model; no training"
echo "FRAMEWORK: accepted Koh Joint CBM with ResNet-50; Standard seed 1"
echo "DATA: 90 rows selected from the accepted 5,000-row controlled-swap population"
echo "COVERAGE: every exact inserted value of tail, wing, beak, foot, and eye"
echo "MEASURES: raw movement, unchanged-part damage, saved-head class evidence, intended swap response"
echo "INPUT: $SWAP_ROOT/funnybirds-cbm-s1.csv"
echo "CHECKPOINT: $CHECKPOINT"
echo "OUTPUT: $OUT"
echo "DEPENDENCY: none"
echo "TIME LIMIT: 01:00:00"
echo "RESULT STATUS: small mechanism preflight; expand only after all cells are reviewed"
echo "COMMAND: sbatch --job-name=fb_xpart_pre --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,OUT=$OUT,ROWS_PER_INPUT_PART=18,CHECKPOINT=$CHECKPOINT,SWAP_ROOT=$SWAP_ROOT curated/train/funnybird_allpart_swap_preflight_job.slurm"

job_id=$(sbatch --parsable --job-name=fb_xpart_pre \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,OUT=$OUT,ROWS_PER_INPUT_PART=18,CHECKPOINT=$CHECKPOINT,SWAP_ROOT=$SWAP_ROOT" \
  "$REPO/curated/train/funnybird_allpart_swap_preflight_job.slurm")
echo "SUBMITTED: job=$job_id output=$OUT"
scontrol show job -dd "$job_id" | grep -E \
  'JobId=|JobName=|JobState=|Reason=|Command=|WorkDir=|StdOut=|TimeLimit=|Environment='
