#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=fb_feats_rl
#SBATCH --output=logs/fb_feats_rl_%j.out

# Extract backbone features for all 5 relabeled MCBM checkpoints.
# Required before run_z_ordering_sweep_rl_adroit.sh can run.
# Outputs: features/resnet50_mcbm_fb_gamma{G}_rl/{layer}_{split}.pt

set -e
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cubvision-gpu

FB_ROOT="data/FunnyBirds"

for GAMMA in 0.0 0.1 0.5 1.0 5.0; do
    CKPT="checkpoints_funnybirds/mcbm_fb_gamma${GAMMA}_rl.pth"
    FEAT_DIR="features/resnet50_mcbm_fb_gamma${GAMMA}_rl"
    echo "[feats] gamma=${GAMMA}  ckpt=${CKPT}  out=${FEAT_DIR}"
    python -m scripts.extract_features_funnybirds \
        --funnybirds_root "${FB_ROOT}" \
        --checkpoint      "${CKPT}" \
        --features_dir    "${FEAT_DIR}" \
        --batch_size 64 \
        --num_workers 4
done

echo "[feats] All done: $(date)"
