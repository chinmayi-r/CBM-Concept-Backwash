#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=curated_cbm_cub
#SBATCH --output=logs/curated_cbm_cub_%A.out
#
# Train CBM on CUB-200, all four regimes, via the official experiments.py
# (yewsiang/ConceptBottleneck -- note: official entrypoint is repo-ROOT
# experiments.py, not src/experiments.py). Commands mirror
# external/ConceptBottleneck/CUB/README.md verbatim (n_attributes=112,
# data_dir class_attr_data_10). Mirrors run_cub_train_cbm.sh's Adroit
# conventions; run inside the `cbm` env built by curated/setup.sh (NOT
# cubvision-gpu -- CBM/MCBM have separate envs, see curated/README.md).
set -euo pipefail
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cbm

: "${CURATED_DATA:?set CURATED_DATA to your data root}"
SEED="${1:-1}"
CBM="external/ConceptBottleneck"
DATA="$CURATED_DATA/CUB_processed/class_attr_data_10"
OUT="$CURATED_DATA/runs/cub_cbm_seed${SEED}"
mkdir -p "$OUT"
cd "$CBM"

echo "### [1/4] Concept x->c (shared by Independent & Sequential)"
python3 experiments.py cub Concept_XtoC --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/ConceptModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -n_attributes 112 -normalize_loss \
  -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck

echo "### [2/4] Independent c->y (label head sees GT concepts)"
python3 experiments.py cub Independent_CtoY --seed "$SEED" \
  -log_dir "$OUT/IndependentModel/outputs/" -e 500 -optimizer sgd \
  -use_attr -data_dir "$DATA" -n_attributes 112 -no_img \
  -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000

echo "### [3/4] Sequential c->y (on frozen x->c predictions)"
# Requires the predicted-concept data dir produced from the x->c model; see CUB/README.md.
python3 experiments.py cub Sequential_CtoY --seed "$SEED" \
  -log_dir "$OUT/SequentialModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr \
  -data_dir "$CURATED_DATA/runs/ConceptModel_PredConcepts" -n_attributes 112 \
  -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000

echo "### [4/4] Joint end-to-end (attr_loss_weight=0.01)"
python3 experiments.py cub Joint --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/JointModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -n_attributes 112 -attr_loss_weight 0.01 \
  -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 \
  -scheduler_step 1000 -end2end

echo "Done -> $OUT"
