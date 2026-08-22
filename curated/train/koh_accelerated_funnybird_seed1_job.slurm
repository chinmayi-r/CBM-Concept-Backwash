#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=koh_accelerated_seed1_%x_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-cubvision-gpu}"
cd "${REPO:?set REPO}"

backup_restart() {
  if [ -n "${KOH_RESTART_BACKUP_DIR:-}" ]; then
    mkdir -p "$KOH_RESTART_BACKUP_DIR"
    find "${KOH_OUTPUT_ROOT:?}/funnybirds/standard/seed1" -maxdepth 1 \
      -name 'restart_state.pth' -type f -exec cp -p {} "$KOH_RESTART_BACKUP_DIR/" \; \
      2>/dev/null || true
  fi
}
on_interrupt() {
  backup_restart
  exit 143
}
trap on_interrupt USR1 TERM

python3 curated/analysis/test_koh_accelerated_restart.py
export KOH_ACCELERATED_RUN_APPROVED=YES
bash curated/train/run_koh_accelerated_funnybird_seed1.sh
