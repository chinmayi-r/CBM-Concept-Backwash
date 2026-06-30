#!/usr/bin/env bash
# Train CBM restricted to the first 70 CUB classes (prof note #3), so every test
# image has a matching CUB70 mask. Two label variants are supported:
#   original  -> CUB_processed/class_attr_data_10
#   relabeled -> CUB_processed/class_attr_data_10_relabeled (from relabel_cub_with_cub70.py)
# so prof note #4 can compare grounding before/after label cleaning.
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-1}"
LABELS="${2:-original}"   # original | relabeled
CBM="curated/external/ConceptBottleneck"

case "$LABELS" in
  original)  SRC="$CURATED_DATA/CUB_processed/class_attr_data_10" ;;
  relabeled) SRC="$CURATED_DATA/CUB_processed/class_attr_data_10_relabeled" ;;
  *) echo "labels must be original|relabeled" >&2; exit 1 ;;
esac
FILT="$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_${LABELS}"
OUT="$CURATED_DATA/runs/cub70_cbm_${LABELS}_seed${SEED}"
mkdir -p "$FILT" "$OUT"

echo "### Filter pickles to classes 0..69"
python3 - "$SRC" "$FILT" <<'PY'
import pickle, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for split in ("train", "val", "test"):
    f = src / f"{split}.pkl"
    if not f.exists():
        continue
    recs = [r for r in pickle.loads(f.read_bytes()) if r["class_label"] < 70]
    (dst / f"{split}.pkl").write_bytes(pickle.dumps(recs))
    print(f"  {split}: {len(recs)} records")
PY

cd "$CBM"
echo "### x->c on CUB70 (verify -num_classes flag name against experiments.py --help)"
python3 src/experiments.py cub Concept_XtoC --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/ConceptModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$FILT" -n_attributes 112 -num_classes 70 \
  -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 \
  -scheduler_step 1000 -bottleneck

echo "### Independent c->y on CUB70"
python3 src/experiments.py cub Independent_CtoY --seed "$SEED" \
  -log_dir "$OUT/IndependentModel/outputs/" -e 500 -optimizer sgd \
  -use_attr -data_dir "$FILT" -n_attributes 112 -num_classes 70 -no_img \
  -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000

echo "Done -> $OUT"
