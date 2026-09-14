#!/usr/bin/env bash
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
: "${STANDARD_ALLPART_OUT:?export STANDARD_ALLPART_OUT to the completed Standard all-part preflight directory}"

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RUN_ID="rlv2_matched_allpart_swap_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$CURATED_DATA/funnybird_allpart_preflight_v1/$RUN_ID"
MODEL_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/rlv2/seed1"
CHECKPOINT="$MODEL_ROOT/final_model_1.pth"
MODEL_MANIFEST="$MODEL_ROOT/SUCCESS.json"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
SELECTION_CSV="$STANDARD_ALLPART_OUT/selected_swap_rows.csv"

test -s "$SELECTION_CSV" || { echo "ERROR: missing Standard matched selection: $SELECTION_CSV" >&2; exit 2; }
git -C "$REPO" diff --quiet -- \
  curated/analysis/funnybird_allpart_swap_preflight.py \
  curated/analysis/funnybird_allpart_swap_preflight_core.py \
  curated/analysis/test_funnybird_allpart_swap_preflight.py \
  curated/train/funnybird_allpart_swap_preflight_job.slurm \
  curated/train/submit_funnybird_rlv2_allpart_swap_preflight.sh || {
    echo "ERROR: the RLv2 all-part preflight source differs from the checked-out commit" >&2
    exit 2
  }

echo "===== CURRENT CLUSTER STATE ====="
squeue -u "$USER" -o "%.18i %.30j %.2t %.12M %.12l %R"
echo "===== RECENT MATCHING JOBS ====="
sacct -u "$USER" --starttime now-7days -X \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End |
  grep -E 'JobID|fb_xpart_rl' || true

echo "===== PROPOSED RLV2 PREFLIGHT 2: MATCHED ALL-TO-ALL SWAPS ====="
echo "QUESTION: after visibility-aware relabeling, which cross-part pathways remain on the same pixels?"
echo "PREDICTION: label conflict as the cause predicts smaller off-diagonal tail damage and context use"
echo "ACTION: replay the exact Standard 90 render IDs through frozen RLv2; no training or fitted probe"
echo "PARITY: render ID, input part, values, species IDs, and both RGB hashes must match Standard"
echo "MODEL: accepted Koh Joint ResNet-50 RLv2 seed 1"
echo "CHECKPOINT: $CHECKPOINT"
echo "SWAPS: $SWAP_ROOT/funnybirds-cbm-rlv2matched-s1.csv"
echo "STANDARD SELECTION: $SELECTION_CSV"
echo "OUTPUT: $OUT"
echo "DEPENDENCY: none"
echo "TIME LIMIT: 01:00:00"
echo "COMMAND: sbatch --job-name=fb_xpart_rl --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,OUT=$OUT,REGIME_LABEL=RLv2,CSV_NAME=funnybirds-cbm-rlv2matched-s1.csv,SELECTION_CSV=$SELECTION_CSV,CHECKPOINT=$CHECKPOINT,MODEL_MANIFEST=$MODEL_MANIFEST,SWAP_ROOT=$SWAP_ROOT curated/train/funnybird_allpart_swap_preflight_job.slurm"

job_id=$(sbatch --parsable --job-name=fb_xpart_rl \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,OUT=$OUT,REGIME_LABEL=RLv2,CSV_NAME=funnybirds-cbm-rlv2matched-s1.csv,SELECTION_CSV=$SELECTION_CSV,CHECKPOINT=$CHECKPOINT,MODEL_MANIFEST=$MODEL_MANIFEST,SWAP_ROOT=$SWAP_ROOT" \
  "$REPO/curated/train/funnybird_allpart_swap_preflight_job.slurm")
echo "SUBMITTED: job=$job_id output=$OUT"
scontrol show job -dd "$job_id" | grep -E \
  'JobId=|JobName=|JobState=|Reason=|Command=|WorkDir=|StdOut=|TimeLimit=|Environment='
