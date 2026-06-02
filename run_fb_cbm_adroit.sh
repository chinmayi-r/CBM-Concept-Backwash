#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=06:00:00           # 6 hours
#SBATCH --job-name=fb_cbm
#SBATCH --output=logs/fb_cbm_%j.out

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_funnybirds

module load anaconda3/2025.6
conda activate cubvision-gpu

echo "[fb_cbm] Training CBM on FunnyBirds (50 classes, 26 concepts)"

python -m scripts.train_cbm_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --checkpoint_dir  checkpoints_funnybirds \
    --backbone_ckpt   checkpoints_funnybirds/resnet50_funnybirds_best.pth \
    --epochs_stage1   12 \
    --epochs_stage2   10 \
    --batch_size      64 \
    --lr              1e-3 \
    --lambda_c        1.0 \
    --num_workers     4 \
    --device          cuda

echo "[fb_cbm] Done. Checkpoint: checkpoints_funnybirds/cbm_funnybirds.pth"

echo "[fb_cbm] Extracting backbone features..."

python -m scripts.extract_features_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --checkpoint      checkpoints_funnybirds/cbm_funnybirds.pth \
    --features_dir    features/resnet50_cbm_funnybirds \
    --batch_size      64 \
    --num_workers     4

echo "[fb_cbm] Features saved to features/resnet50_cbm_funnybirds"
