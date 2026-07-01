#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=curated_mcbm_cub70
#SBATCH --output=logs/curated_mcbm_cub70_%A_%a.out
#SBATCH --array=0-7               # one job per gamma value
#
# Train MCBM on the 70-class CUB70 subset (prof notes #3/#4). LABELS=original|
# relabeled selects which pkls get class-filtered (original CUB_processed/
# class_attr_data_10, or the relabel_cub_with_cub70.py output) -- mirrors
# cbm_cub70.sh's own filtering step so both frameworks train on identical
# CUB70 splits.
#
# Same config-template/sed-render mechanism as mcbm_cub.sh (see that file and
# curated/README.md's "narrow, documented exception" note): __CURATED_DATA__/
# __LABELS__/__GAMMA__ are substituted into
# external/minimal_cbm/configs/cub70/cub70-mcbm.yaml before each run.
set -euo pipefail
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate mcbm

: "${CURATED_DATA:?set CURATED_DATA}"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA
SEED="${1:-42}"
LABELS="${2:-original}"   # original | relabeled
MCBM="external/minimal_cbm"
TEMPLATE="train/configs/cub70-mcbm.yaml"

case "$LABELS" in
  original)  SRC="$CURATED_DATA/CUB_processed/class_attr_data_10" ;;
  relabeled) SRC="$CURATED_DATA/CUB_processed/class_attr_data_10_relabeled" ;;
  *) echo "labels must be original|relabeled" >&2; exit 1 ;;
esac
FILT="$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_${LABELS}"
mkdir -p "$FILT"

echo "### Filter pickles to classes 0..69 (labels=$LABELS)"
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

GAMMAS=(0.0 0.05 0.1 0.2 0.5 1.0 2.5 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$MCBM/configs/cub70"
sed -e "s#__CURATED_DATA__#${CURATED_DATA}#g" -e "s/__GAMMA__/${GAMMA}/g" \
    -e "s/__LABELS__/${LABELS}/g" \
  "$TEMPLATE" > "$MCBM/configs/cub70/cub70-mcbm.yaml"

cd "$MCBM"
echo "### MCBM CUB70  labels=${LABELS}  gamma=${GAMMA}  seed=$SEED"
python3 bin/train.py cub70-mcbm -s "$SEED"
echo "Done. labels=${LABELS} gamma=${GAMMA}"
