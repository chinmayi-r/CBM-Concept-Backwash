#!/usr/bin/env bash
# Train CBM on FunnyBirds via the official trainer, using the pickled lists from
# build_funnybirds_cbm_data.py. FunnyBirds is not in the CBM paper, so this is an
# adaptation: same trainer, same three-regime structure, our concept count.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-1}"
CBM="curated/external/ConceptBottleneck"
DATA="$CURATED_DATA/funnybirds_processed"
OUT="$CURATED_DATA/runs/funnybirds_cbm_seed${SEED}"
# N_ATTR must equal len(funnybirds_concepts.flat_concept_names()); N_CLASSES = #FunnyBirds classes
N_ATTR="${N_ATTR:-23}"
N_CLASSES="${N_CLASSES:-50}"
mkdir -p "$OUT"
cd "$CBM"

echo "### x->c (concept) on FunnyBirds  [n_attributes=$N_ATTR, num_classes=$N_CLASSES]"
python3 src/experiments.py cub Concept_XtoC --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/ConceptModel/outputs/" -e 500 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -n_attributes "$N_ATTR" -num_classes "$N_CLASSES" \
  -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 \
  -scheduler_step 500 -bottleneck

echo "### Independent c->y on FunnyBirds"
python3 src/experiments.py cub Independent_CtoY --seed "$SEED" \
  -log_dir "$OUT/IndependentModel/outputs/" -e 300 -optimizer sgd \
  -use_attr -data_dir "$DATA" -n_attributes "$N_ATTR" -num_classes "$N_CLASSES" \
  -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 500

echo "### Joint end-to-end on FunnyBirds"
python3 src/experiments.py cub Joint --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/JointModel/outputs/" -e 500 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$DATA" -n_attributes "$N_ATTR" -num_classes "$N_CLASSES" \
  -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 \
  -lr 0.001 -scheduler_step 500 -end2end

echo "Done -> $OUT"
