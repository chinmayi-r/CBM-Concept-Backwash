#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=mcbm_seeded_%x_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate cubvision-gpu
cd "${REPO:?set REPO}"

retry_timeout_only() {
  count="${SLURM_RESTART_COUNT:-0}"
  if [ "$count" -lt 2 ]; then
    echo "[INFRASTRUCTURE RETRY] requeue $SLURM_JOB_ID; prior restarts=$count"
    scontrol requeue "$SLURM_JOB_ID"
  else
    echo "ERROR: infrastructure retry limit reached" >&2
  fi
  exit 143
}
trap retry_timeout_only USR1

bash curated/train/mcbm_seeded_stage.sh
