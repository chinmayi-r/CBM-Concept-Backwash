#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=04:00:00           # 4 hours
#SBATCH --job-name=mcbm_feats
#SBATCH --output=logs/mcbm_feats_%A_%a.out
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

echo "[mcbm_feats] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"

python scripts/extract_features_mcbm.py \
    --cub_root      data/CUB_200_2011 \
    --checkpoint    "checkpoints_mcbm/mcbm_gamma${GAMMA}.pth" \
    --features_dir  "features/resnet50_mcbm_gamma${GAMMA}" \
    --batch_size    64 \
    --num_workers   4
