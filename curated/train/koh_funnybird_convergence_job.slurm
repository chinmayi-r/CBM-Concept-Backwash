#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=koh_convergence_%x_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-cubvision-gpu}"
cd "${REPO:?set REPO}"

on_usr1() {
  count="${SLURM_RESTART_COUNT:-0}"
  if [ "$count" -lt 2 ]; then
    echo "[INFRASTRUCTURE RETRY] requeueing job=$SLURM_JOB_ID completed_requeues=$count"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo "ERROR: infrastructure retry limit reached after $count requeues" >&2
  exit 143
}
trap on_usr1 USR1

python3 curated/analysis/test_koh_accelerated_restart.py
bash curated/train/koh_funnybird_convergence_stage.sh
