#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=koh_accelerated_seed1_%x_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-cubvision-gpu}"
cd "${REPO:?set REPO}"

backup_restart() {
  if [ -n "${KOH_RESTART_BACKUP_DIR:-}" ]; then
    mkdir -p "$KOH_RESTART_BACKUP_DIR"
    find "${KOH_OUTPUT_ROOT:?}/funnybirds/${LABELS:-standard}/seed1" -maxdepth 1 \
      -name 'restart_state.pth' -type f -exec cp -p {} "$KOH_RESTART_BACKUP_DIR/" \; \
      2>/dev/null || true
  fi
}
on_usr1() {
  backup_restart
  restart_count="${SLURM_RESTART_COUNT:-0}"
  if [ "$restart_count" -lt 2 ]; then
    echo "[INFRASTRUCTURE RETRY] requeueing job=$SLURM_JOB_ID completed_requeues=$restart_count"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo "ERROR: infrastructure retry limit reached after $restart_count requeues" >&2
  exit 143
}
on_term() {
  backup_restart
  exit 143
}
trap on_usr1 USR1
trap on_term TERM

python3 curated/analysis/test_koh_accelerated_restart.py
export DATASET=funnybirds
export LABELS="${LABELS:-standard}"
case "$LABELS" in standard|rlv2) ;; *) echo "ERROR: invalid labels=$LABELS" >&2; exit 2 ;; esac
export SEED=1
export BACKBONE=resnet50
export KOH_TRAINING_PROTOCOL=accelerated_v1
export KOH_OUTPUT_ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_resnet_accelerated_v1}"
export KOH_RESTART_BACKUP_DIR="${KOH_RESTART_BACKUP_DIR:-$CURATED_DATA/koh_joint_resnet_accelerated_restart_backup/funnybirds/$LABELS/seed1}"
bash curated/train/koh_joint_stage.sh
