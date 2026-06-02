#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=08:00:00           # 8 hours
#SBATCH --job-name=fb_resnet
#SBATCH --output=logs/fb_resnet_%j.out

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_funnybirds

module load anaconda3/2025.6
conda activate cubvision-gpu

echo "[fb_resnet] Training ResNet-50 on FunnyBirds (50 classes)"

python -m scripts.train_resnet_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --out_ckpt        checkpoints_funnybirds/resnet50_funnybirds_best.pth \
    --epochs          50 \
    --batch_size      64 \
    --lr              1e-4 \
    --num_workers     4

echo "[fb_resnet] Done."
