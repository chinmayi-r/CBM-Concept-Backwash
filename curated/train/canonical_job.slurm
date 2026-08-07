#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=canonical_%x_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate cubvision-gpu
cd "${REPO:?set REPO}"

# Only a scheduler time-limit warning is automatically retried. Ordinary exit
# code 1 is treated as deterministic and is left visible for correction.
on_timeout() {
  local count="${SLURM_RESTART_COUNT:-0}"
  if [ "$count" -lt "${MAX_REQUEUES:-2}" ]; then
    echo "[TRANSIENT TIME LIMIT] requeue $SLURM_JOB_ID; prior restarts=$count"
    scontrol requeue "$SLURM_JOB_ID"
  else
    echo "[ERROR] retry limit reached for $SLURM_JOB_ID" >&2
  fi
  exit 143
}
trap on_timeout USR1

bash curated/train/canonical_stage.sh
