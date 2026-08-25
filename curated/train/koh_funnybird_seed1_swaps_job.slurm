#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=koh_fb_seed1_swaps_%j.out

set -euo pipefail
module load anaconda3/2025.6
conda activate "${CONDA_ENV:-cubvision-gpu}"
cd "${REPO:?set REPO}"
bash curated/train/koh_funnybird_seed1_swaps_stage.sh
