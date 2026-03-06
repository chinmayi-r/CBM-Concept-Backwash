#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=04:00:00           # 4 hours
#SBATCH --job-name=cub_probes
#SBATCH --output=logs/cub_probes_%j.out

# Safer execution
set -e
set -x

# Use the folder from which sbatch is invoked
cd "$SLURM_SUBMIT_DIR"

# Ensure required directory structure
mkdir -p logs
mkdir -p checkpoints
mkdir -p features
mkdir -p results/probes

# Load conda
module load anaconda3/2025.6
# If you need:
# source ~/.bashrc
conda activate cubvision-gpu

# Run probe training module
python -m scripts.train_probes \
    --features_dir features/resnet50_cub \
    --cub_root data/CUB_200_2011 \
    --out_json results/probes/resnet50_cub_probes.json \
    --batch_size 256 \
    --epochs 20
