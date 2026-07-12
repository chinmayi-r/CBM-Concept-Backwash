#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=fb_probes
#SBATCH --output=logs/fb_probes_%j.out

# Run train_probes_funnybirds.py for both baseline and CBM features.
# Produces:
#   results/probes/resnet50_funnybirds_probes_fine.json
#   results/probes/resnet50_funnybirds_cbm_probes_fine.json
#
# Prerequisites (run once on Adroit before this):
#   sbatch run_fb_resnet_adroit.sh        -> checkpoints_funnybirds/resnet50_funnybirds_best.pth
#   sbatch run_fb_cbm_adroit.sh           -> checkpoints_funnybirds/resnet50_cbm_funnybirds_best.pth
#   sbatch run_fb_mcbm_extract_adroit.sh  -> features/resnet50_funnybirds/  +  features/resnet50_cbm_funnybirds/
#                                           (use --plain_resnet flag for baseline)
# Also run prepare_funnybirds_metadata.py once if not already done:
#   python -m scripts.prepare_funnybirds_metadata --funnybirds_root data/FunnyBirds

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results/probes

module load anaconda3/2025.6
conda activate cubvision-gpu

FB_ROOT="data/FunnyBirds"
EPOCHS=15
BATCH=256

# ── Baseline (plain ResNet-50, no concept supervision) ───────────────────────
echo "[fb_probes] Baseline probes"
python -m scripts.train_probes_funnybirds \
    --features_dir    features/resnet50_funnybirds \
    --funnybirds_root ${FB_ROOT} \
    --out_json        results/probes/resnet50_funnybirds_probes_fine.json \
    --epochs          ${EPOCHS} \
    --batch_size      ${BATCH}

# ── CBM backbone ──────────────────────────────────────────────────────────────
echo "[fb_probes] CBM probes"
python -m scripts.train_probes_funnybirds \
    --features_dir    features/resnet50_cbm_funnybirds \
    --funnybirds_root ${FB_ROOT} \
    --out_json        results/probes/resnet50_funnybirds_cbm_probes_fine.json \
    --epochs          ${EPOCHS} \
    --batch_size      ${BATCH}

echo "[fb_probes] Done. Results in results/probes/"