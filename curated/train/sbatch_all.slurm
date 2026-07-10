#!/usr/bin/env bash
#SBATCH --job-name=fb_spine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#
# Fire-and-forget the whole FunnyBirds spine (baselines + MCBM gamma sweep +
# grounding + backwash-vs-gamma plot). Idempotent: if it hits the wall, just
# resubmit the identical command and it resumes where it left off.
#
# SUBMIT FROM curated/ with CURATED_DATA exported:
#   export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
#   GAMMAS="0 1 3 10 30" SEEDS="1" sbatch train/sbatch_all.slurm
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA before sbatch}"
CONDA_ENV="${CONDA_ENV:-cubvision-gpu}"
CURATED="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -d "$CURATED/train" ] || CURATED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# conda/module init reference unset vars; disable -u around activation
set +u
if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then source "$CONDA_SH"
else
  command -v conda >/dev/null 2>&1 || module load anaconda3 2>/dev/null || module load anaconda3/2025.6 2>/dev/null || true
  base=""; command -v conda >/dev/null 2>&1 && base="$(conda info --base)"
  [ -z "$base" ] && [ -n "${CONDA_EXE:-}" ] && base="$(dirname "$(dirname "$CONDA_EXE")")"
  for c in "$base" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -n "$c" ] && [ -f "$c/etc/profile.d/conda.sh" ] && { source "$c/etc/profile.d/conda.sh"; break; }
  done
fi
conda activate "$CONDA_ENV"
set -u

export WANDB_MODE=offline WANDB_DISABLED=true
cd "$CURATED"
bash train/run_all_funnybirds.sh
echo "sbatch_all payload done."
