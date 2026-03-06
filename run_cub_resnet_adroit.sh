#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=04:00:00           # 4 hours
#SBATCH --job-name=cub_feats
#SBATCH --output=logs/cub_feats_%j.out

# Safer execution behavior
set -e   # exit on any error
set -x   # print commands before executing

# Work in directory where sbatch was invoked
cd "$SLURM_SUBMIT_DIR"

# Ensure required dirs exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p features
mkdir -p results/probes

# Load Python / conda environment
module load anaconda3/2025.6
# If non-interactive conda is an issue:
# source ~/.bashrc
conda activate cubvision-gpu

# Run the feature extraction module
python -m scripts.extract_features \
    --cub_root data/CUB_200_2011 \
    --ckpt_path checkpoints/resnet50_cub_best.pth \
    --arch resnet50 \
    --out_dir features/resnet50_cub \
    --batch_size 64
