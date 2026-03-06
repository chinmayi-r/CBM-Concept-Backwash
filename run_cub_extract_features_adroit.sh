#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=08:00:00           # 8 hours
#SBATCH --job-name=cub_resnet
#SBATCH --output=logs/cub_resnet_%j.out

# Make debugging easier:
set -e  # exit on first error
set -x  # print each command before running it

# Always start in the directory where you ran `sbatch`
cd "$SLURM_SUBMIT_DIR"

# Make sure logs and project dirs exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p features
mkdir -p results/probes

# Load Python / conda
module load anaconda3/2025.6

# If 'conda' isn't defined in non-interactive shells, you may need:
# source ~/.bashrc
# or wherever your conda init lives

conda activate cubvision-gpu

# Now run training FROM THIS DIRECTORY.
# Assumptions:
#   - you run `sbatch` from your project root
#   - scripts/train_resnet.py exists relative to that root
#   - CUB is at data/CUB_200_2011/CUB_200_2011

python scripts/train_resnet.py \
  --cub_root data/CUB_200_2011/CUB_200_2011 \
  --arch resnet50 \
  --batch_size 64 \
  --epochs 100 \
  --lr 0.0005 \
  --weight_decay 1e-4 \
  --out_ckpt checkpoints/resnet50_cub_best.pth

