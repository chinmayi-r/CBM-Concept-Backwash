#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=6         # 6 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM
#SBATCH --time=02:00:00           # 2 hours
#SBATCH --job-name=fb_mcbm_feats_fix
#SBATCH --output=logs/fb_mcbm_feats_fix_%A_%a.out
#SBATCH --array=0-4               # one job per gamma value

# Feature extraction for fixed MCBM checkpoints (_fix suffix).
# Reads:  checkpoints_funnybirds/mcbm_fb_gamma{G}_fix.pth
# Writes: features/resnet50_mcbm_fb_gamma{G}_fix/
# Run after run_fb_mcbm_train_fix_adroit.sh completes.

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p features

module load anaconda3/2025.6
conda activate cubvision-gpu

GAMMAS=(0.0 0.1 0.5 1.0 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "[fb_mcbm_feats_fix] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"

python -m scripts.extract_features_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --checkpoint      "checkpoints_funnybirds/mcbm_fb_gamma${GAMMA}_fix.pth" \
    --features_dir    "features/resnet50_mcbm_fb_gamma${GAMMA}_fix" \
    --batch_size      256 \
    --num_workers     6

echo "[fb_mcbm_feats_fix] Done. gamma=${GAMMA}  features=features/resnet50_mcbm_fb_gamma${GAMMA}_fix"
