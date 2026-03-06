#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=06:00:00           # 6 hours (CBM takes longer!)
#SBATCH --job-name=cub_cbm
#SBATCH --output=logs/cub_cbm_%j.out

set -e    # stop on error
set -x    # print commands before running

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_cbm
mkdir -p results
mkdir -p results/probes

module load anaconda3/2025.6
conda activate cubvision-gpu

# ---- TRAIN CBM MODEL ----
python -m scripts.train_cbm_attributes \
    --cub_root data/CUB_200_2011 \
    --output_dir checkpoints_cbm \
    --epochs_concepts 12 \
    --epochs_labels 10 \
    --batch_size 64 \
    --lr_concepts 1e-3 \
    --lr_labels 1e-3 \
    --device cuda
