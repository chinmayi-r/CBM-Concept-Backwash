#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=04:00:00           # 4 hours
#SBATCH --job-name=fb_mcbm_feats
#SBATCH --output=logs/fb_mcbm_feats_%A_%a.out
#SBATCH --array=0-4               # one job per gamma value

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p features

module load anaconda3/2025.6
conda activate cubvision-gpu

# gamma sweep: mirrors run_mcbm_extract_adroit.sh exactly
GAMMAS=(0.0 0.1 0.5 1.0 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "[fb_mcbm_feats] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"

python -m scripts.extract_features_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --checkpoint      "checkpoints_funnybirds/mcbm_fb_gamma${GAMMA}.pth" \
    --features_dir    "features/resnet50_mcbm_fb_gamma${GAMMA}" \
    --batch_size      64 \
    --num_workers     4

echo "[fb_mcbm_feats] Done. gamma=${GAMMA}"
