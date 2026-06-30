#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=curated_cbm_fb
#SBATCH --output=logs/curated_cbm_fb_%A.out
#
# Train CBM on FunnyBirds via curated/patches/run_cbm_funnybirds.py, using the
# pickled lists from build_funnybirds_cbm_data.py. The official CBM trainer
# hardcodes N_CLASSES=200 as a CUB/config.py module constant, imported by
# CUB/train.py at import time -- run_cbm_funnybirds.py patches it to 50
# before CUB.train is ever imported (see that file's docstring). There is no
# `-num_classes` CLI flag in the official parser; `-n_attributes` IS a real
# flag (default N_ATTRIBUTES, overridden here to 26).
# FunnyBirds is not in the CBM paper, so this is an adaptation: same trainer,
# same three-regime structure, our concept/class counts.
set -euo pipefail
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cbm

: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-1}"
CBM="external/ConceptBottleneck"
DATA="$CURATED_DATA/funnybirds_processed"
OUT="$CURATED_DATA/runs/funnybirds_cbm_seed${SEED}"
IMAGE_DIR="$DATA/CUB_200_2011"   # symlink -> funnybirds_root, created by build_funnybirds_cbm_data.py
N_ATTR="${N_ATTR:-26}"
mkdir -p "$OUT"
cd "$CBM"

echo "### x->c (concept) on FunnyBirds  [n_attributes=$N_ATTR, n_classes=50]"
python3 ../../patches/run_cbm_funnybirds.py cub Concept_XtoC --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/ConceptModel/outputs/" -e 500 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -image_dir "$IMAGE_DIR" -n_attributes "$N_ATTR" \
  -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 \
  -scheduler_step 500 -bottleneck

echo "### Independent c->y on FunnyBirds"
python3 ../../patches/run_cbm_funnybirds.py cub Independent_CtoY --seed "$SEED" \
  -log_dir "$OUT/IndependentModel/outputs/" -e 300 -optimizer sgd \
  -use_attr -data_dir "$DATA" -n_attributes "$N_ATTR" \
  -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 500

echo "### Joint end-to-end on FunnyBirds"
python3 ../../patches/run_cbm_funnybirds.py cub Joint --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/JointModel/outputs/" -e 500 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -image_dir "$IMAGE_DIR" -n_attributes "$N_ATTR" \
  -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 \
  -lr 0.001 -scheduler_step 500 -end2end

echo "Done -> $OUT"
