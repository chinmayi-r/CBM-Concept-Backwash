#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=koh_fb_provisional_swaps_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-cubvision-gpu}"
cd "${REPO:?set REPO}"
bash curated/train/koh_funnybird_provisional_swaps_stage.sh
