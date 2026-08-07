#!/usr/bin/env bash
# Submit independent MCBM cells for one dataset/label stage.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
DATASET="${1:?usage: submit_mcbm_seeded_stage.sh DATASET LABELS [SEEDS] [GAMMAS]}"
LABELS="${2:?usage: submit_mcbm_seeded_stage.sh DATASET LABELS [SEEDS] [GAMMAS]}"
SEEDS="${3:-1}"
GAMMAS="${4:-0 0.1 0.3 1 3 5}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"

case "$DATASET:$LABELS" in
  funnybirds:standard|funnybirds:rlv2|cub70:standard|cub:standard) ;;
  *) echo "ERROR: unsupported stage $DATASET:$LABELS" >&2; exit 2 ;;
esac
echo "===== FRESH USER QUEUE ====="
if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
  echo "[DRY RUN] queue lookup and patch application omitted"
else
  squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"
  bash "$REPO/curated/train/verify_canonical_sources.sh"
fi

for seed in $SEEDS; do
  case "$seed" in 1|2|3) ;; *) echo "ERROR: invalid seed $seed" >&2; exit 2 ;; esac
  for gamma in $GAMMAS; do
    case "$gamma" in 0|0.1|0.3|1|3|5) ;; *) echo "ERROR: invalid gamma $gamma" >&2; exit 2 ;; esac
    tag="${gamma//./p}"
    if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
      jid="DRY_m_${DATASET}_${LABELS}_g${tag}_s${seed}"
    else
      jid=$(sbatch --parsable --job-name="m_${DATASET}_${LABELS}_g${tag}_s${seed}" \
        --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=$DATASET,LABELS=$LABELS,GAMMA=$gamma,SEED=$seed" \
        "$REPO/curated/train/mcbm_seeded_job.slurm")
    fi
    echo "$DATASET $LABELS gamma=$gamma seed=$seed job=$jid"
  done
done
