#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=8         # 8 CPU cores (feeds 6 DataLoader workers + main + spare)
#SBATCH --mem=48G                 # 48 GB RAM (batch_size=128 + AMP overhead)
#SBATCH --time=06:00:00           # 6 hours (AMP ~1.5x faster than baseline)
#SBATCH --job-name=fb_mcbm_fix
#SBATCH --output=logs/fb_mcbm_fix_%A_%a.out
#SBATCH --array=0-4               # one job per gamma value

# Fixed MCBM training: BCE and IB penalty on clean z (not noisy z_s),
# and IB penalty uses concept-label targets (+3/-3) instead of self-consistent q(z).
# Saves to checkpoints_funnybirds/mcbm_fb_gamma{G}_fix.pth
# Compare against original (buggy) mcbm_fb_gamma{G}.pth using the sweep.

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_funnybirds

module load anaconda3/2025.6
conda activate cubvision-gpu

GAMMAS=(0.0 0.1 0.5 1.0 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "[fb_mcbm_fix] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"

python -m scripts.train_mcbm_funnybirds \
    --funnybirds_root data/FunnyBirds \
    --checkpoint_dir  checkpoints_funnybirds \
    --backbone_ckpt   checkpoints_funnybirds/resnet50_funnybirds_best.pth \
    --gamma           "${GAMMA}" \
    --epochs_stage1   12 \
    --epochs_stage2   10 \
    --batch_size      128 \
    --lr              1e-3 \
    --sigma           1.0 \
    --lambda_c        1.0 \
    --num_workers     6 \
    --device          cuda \
    --ckpt_suffix     _fix

echo "[fb_mcbm_fix] Done. gamma=${GAMMA}  checkpoint=checkpoints_funnybirds/mcbm_fb_gamma${GAMMA}_fix.pth"
