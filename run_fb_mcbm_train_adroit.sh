#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=08:00:00           # 8 hours
#SBATCH --job-name=fb_mcbm_train
#SBATCH --output=logs/fb_mcbm_train_%A_%a.out
#SBATCH --array=0-4               # one job per gamma value

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_funnybirds

module load anaconda3/2025.6
conda activate cubvision-gpu

# gamma sweep: mirrors run_mcbm_train_adroit.sh exactly
GAMMAS=(0.0 0.1 0.5 1.0 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "[fb_mcbm_train] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"

python -m scripts.train_mcbm_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --checkpoint_dir  checkpoints_funnybirds \
    --backbone_ckpt   checkpoints_funnybirds/resnet50_funnybirds_best.pth \
    --gamma           "${GAMMA}" \
    --epochs_stage1   12 \
    --epochs_stage2   10 \
    --batch_size      64 \
    --lr              1e-3 \
    --sigma           1.0 \
    --lambda_c        1.0 \
    --num_workers     4 \
    --device          cuda

echo "[fb_mcbm_train] Done. gamma=${GAMMA}"
