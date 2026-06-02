#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=fb_mcbm_rl
#SBATCH --output=logs/fb_mcbm_rl_%A_%a.out
#SBATCH --array=0-4               # one job per gamma: 0.0, 0.1, 0.5, 1.0, 5.0

# Train MCBM on FunnyBirds with IMAGE-LEVEL concept labels.
#
# Prerequisite: run run_fb_make_labels_adroit.sh first to generate
#   data/FunnyBirds/concept_labels_image_level.npy
#
# Checkpoints saved to:
#   checkpoints_funnybirds/mcbm_fb_gamma{GAMMA}_rl.pth
#
# To compare original vs relabeled: run the z-ordering sweep for both
# sets of checkpoints (set USE_RL=True in run_z_ordering_sweep.py or
# pass --suffix _rl to distinguish CSV output).

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints_funnybirds

module load anaconda3/2025.6
conda activate cubvision-gpu

GAMMAS=(0.0 0.1 0.5 1.0 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
CONCEPT_LABELS="data/FunnyBirds/concept_labels_image_level.npy"

echo "[fb_mcbm_rl] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  gamma=${GAMMA}"
echo "[fb_mcbm_rl] Using image-level labels: ${CONCEPT_LABELS}"

if [ ! -f "${CONCEPT_LABELS}" ]; then
    echo "[fb_mcbm_rl] ERROR: ${CONCEPT_LABELS} not found."
    echo "[fb_mcbm_rl] Run run_fb_make_labels_adroit.sh first."
    exit 1
fi

python -m scripts.train_mcbm_funnybirds \
    --funnybirds_root  data/FunnyBirds \
    --checkpoint_dir   checkpoints_funnybirds \
    --backbone_ckpt    checkpoints_funnybirds/resnet50_funnybirds_best.pth \
    --gamma            "${GAMMA}" \
    --epochs_stage1    12 \
    --epochs_stage2    10 \
    --batch_size       64 \
    --lr               1e-3 \
    --sigma            1.0 \
    --lambda_c         1.0 \
    --num_workers      4 \
    --device           cuda \
    --concept_labels   "${CONCEPT_LABELS}"

echo "[fb_mcbm_rl] Done. Checkpoint: checkpoints_funnybirds/mcbm_fb_gamma${GAMMA}_rl.pth"
