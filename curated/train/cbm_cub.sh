#!/usr/bin/env bash
# Train CBM on CUB-200, all three regimes, via the official src/experiments.py.
# Commands mirror external/ConceptBottleneck/CUB/README.md verbatim (n_attributes=112,
# data_dir class_attr_data_10). Run inside the `cbm` env.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA to your data root}"
SEED="${1:-1}"
CBM="curated/external/ConceptBottleneck"
DATA="$CURATED_DATA/CUB_processed/class_attr_data_10"
OUT="$CURATED_DATA/runs/cub_cbm_seed${SEED}"
mkdir -p "$OUT"
cd "$CBM"

echo "### [1/4] Concept x->c (shared by Independent & Sequential)"
python3 src/experiments.py cub Concept_XtoC --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/ConceptModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -n_attributes 112 -normalize_loss \
  -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck

echo "### [2/4] Independent c->y (label head sees GT concepts)"
python3 src/experiments.py cub Independent_CtoY --seed "$SEED" \
  -log_dir "$OUT/IndependentModel/outputs/" -e 500 -optimizer sgd \
  -use_attr -data_dir "$DATA" -n_attributes 112 -no_img \
  -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000

echo "### [3/4] Sequential c->y (on frozen x->c predictions)"
# Requires the predicted-concept data dir produced from the x->c model; see CUB/README.md.
python3 src/experiments.py cub Sequential_CtoY --seed "$SEED" \
  -log_dir "$OUT/SequentialModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr \
  -data_dir "$CURATED_DATA/runs/ConceptModel_PredConcepts" -n_attributes 112 \
  -no_img -b 64 -weight_decay 0.00004 -lr 0.001 -scheduler_step 1000

echo "### [4/4] Joint end-to-end (attr_loss_weight=0.01)"
python3 src/experiments.py cub Joint --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/JointModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -n_attributes 112 -attr_loss_weight 0.01 \
  -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 \
  -scheduler_step 1000 -end2end

echo "Done -> $OUT"
