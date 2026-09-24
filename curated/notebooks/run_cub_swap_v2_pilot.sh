#!/usr/bin/env bash
# Small frozen-model pilot. No training and no Slurm submission.
set -euo pipefail
: "${CURATED_DATA:?Set CURATED_DATA to the existing shared data directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$CURATED_DATA/koh_joint_resnet_v1/cub70/standard/seed1"
OUT="${CUB_SWAP_PILOT_OUT:-$CURATED_DATA/cub_swap_v2_pilot_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${CUB_SWAP_DEVICE:-cpu}"
echo "Research step: test whether silhouette-preserving CUB replacements create interpretable donor evidence."
echo "Frozen official Koh Joint ResNet-50; CUB70 Standard seed 1; 112 concepts, 70 species."
echo "Training=no; Slurm submission=no; model-score selection=no; device=$DEVICE"
echo "Stages: dependency and artifact check -> label/mask triplets -> four-image edits -> original replay -> frozen inference -> blinded review."
echo "Output=$OUT; old results remain in their existing roots."
if command -v squeue >/dev/null 2>&1; then
  squeue -u "$USER"
  sacct -j 3357749 --format=JobID,State,ExitCode,Elapsed
fi
python -c 'import torch, torchvision, pandas, numpy, PIL, scipy; print("Dependencies available; torch", torch.__version__, "torchvision", torchvision.__version__)'
for p in "$ROOT/SUCCESS.json" "$ROOT/best_model_1.pth" "$ROOT/final_test.parquet" "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/test.pkl"; do
  test -s "$p" || { echo "ERROR: missing input $p; no computation started" >&2; exit 2; }
  echo "Input=$p"
done
test ! -e "$OUT" || { echo "ERROR: output exists; choose a new CUB_SWAP_PILOT_OUT" >&2; exit 2; }
python analysis/canonical_manifest.py verify --manifest "$ROOT/SUCCESS.json"
echo "PREFLIGHT ONLY - NO SCIENTIFIC TRAINING"
PYTHONPATH=analysis python -m unittest discover -s tests -p 'test_cub_mask_swap_v2.py'
mkdir -p "$OUT"
python analysis/cub_swap_plan.py --evaluation "$ROOT/final_test.parquet" \
  --images "$CURATED_DATA/CUB_200_2011/images" --masks "$CURATED_DATA/cub70/masks" \
  --per-group 3 --out "$OUT/selection"
python analysis/cub_mask_swap_v2.py --plan "$OUT/selection/plan.json" --out "$OUT/edits"
python analysis/cub_swap_frozen_inference.py --edits "$OUT/edits" \
  --manifest "$ROOT/SUCCESS.json" --checkpoint "$ROOT/best_model_1.pth" \
  --evaluation "$ROOT/final_test.parquet" \
  --data-pkl "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/test.pkl" \
  --device "$DEVICE" --out "$OUT/inference"
echo "Pilot computation finished. METHOD NOT CALIBRATED: inspect every edits/*/review.png before reading predictions."
echo "Record explicit checks in $OUT/inference/blinded_review.csv; do not set flags automatically."
echo "Notebook input after review: CUB_INTERVENTION_RESULTS=$OUT/inference"
