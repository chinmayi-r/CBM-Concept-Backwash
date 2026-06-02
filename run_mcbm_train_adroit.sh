#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=08:00:00           # 8 hours
#SBATCH --job-name=mcbm_train
#SBATCH --output=logs/mcbm_train_%A_%a.out
#SBATCH --array=0-4               # one job per gamma value

set -e    # stop on error
set -x    # print commands before running

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_mcbm
mkdir -p features
mkdir -p results/probes

module load anaconda3/2025.6
conda activate cubvision-gpu

# gamma sweep: index maps to gamma value
GAMMAS=(0.0 0.1 0.5 1.0 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "[mcbm_train] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"

python scripts/train_mcbm.py \
    --cub_root          data/CUB_200_2011 \
    --checkpoint_dir    checkpoints_mcbm \
    --backbone_ckpt     checkpoints/resnet50_cub_best.pth \
    --gamma             "${GAMMA}" \
    --epochs_stage1     12 \
    --epochs_stage2     10 \
    --batch_size        64 \
    --lr                1e-3 \
    --sigma             1.0 \
    --lambda_c          1.0 \
    --num_workers       4 \
    --device            cuda
