#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=curated_cbm_cub70
#SBATCH --output=logs/curated_cbm_cub70_%A.out
#
# Train CBM restricted to the first 70 CUB classes (prof note #3), so every
# test image has a matching CUB70 mask. Two label variants are supported:
#   original  -> CUB_processed/class_attr_data_10
#   relabeled -> CUB_processed/class_attr_data_10_relabeled (from relabel_cub_with_cub70.py)
# so prof note #4 can compare grounding before/after label cleaning.
#
# N_CLASSES=70 hits the same CUB/config.py hardcoded-constant problem as
# FunnyBirds (see patches/run_cbm_funnybirds.py's docstring) -- reused here via
# CBM_N_CLASSES/CBM_N_ATTRIBUTES env vars instead of a nonexistent -num_classes
# CLI flag.
set -e    # stop on error (not -u/pipefail: module load/conda activate reference unset vars like $PS1 in a non-interactive batch shell, which -u turns into a hard crash)
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cbm

: "${CURATED_DATA:?set CURATED_DATA}"
SEED="${1:-1}"
LABELS="${2:-original}"   # original | relabeled
CBM="external/ConceptBottleneck"

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

export CBM_N_CLASSES=70 CBM_N_ATTRIBUTES=112
cd "$CBM"
echo "### x->c on CUB70 [labels=$LABELS, n_classes=70]"
python3 ../../patches/run_cbm_funnybirds.py cub Concept_XtoC --seed "$SEED" -ckpt 1 \
  -log_dir "$OUT/ConceptModel/outputs/" -e 1000 -optimizer sgd \
  -pretrained -use_aux -use_attr -weighted_loss multiple \
  -data_dir "$FILT" -n_attributes 112 \
  -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 \
  -scheduler_step 1000 -bottleneck

echo "### Independent c->y on CUB70"
python3 ../../patches/run_cbm_funnybirds.py cub Independent_CtoY --seed "$SEED" \
  -log_dir "$OUT/IndependentModel/outputs/" -e 500 -optimizer sgd \
  -use_attr -data_dir "$FILT" -n_attributes 112 -no_img \
  -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000

echo "Done -> $OUT"
