#!/usr/bin/env bash
# Rebuild and execute Notebook 05 from accepted official-Koh artifacts.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA to the shared curated_data directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATED="$(cd "$HERE/.." && pwd)"
cd "$CURATED"

CUB70_ROOT="$CURATED_DATA/koh_joint_resnet_v1/cub70/standard/seed1"
FB_ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1"
SWAP_ROOT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"

echo "===== NOTEBOOK 05: CUB70 STANDARD CBM ====="
echo "goal=test natural-image warning signs of concept backwash and calibrate matched recall on FunnyBird controlled swaps"
echo "primary model=official Koh Joint ResNet-50 CUB70 Standard seed 1"
echo "dimensions=112 concepts, 70 species"
echo "CUB operation=observational mask visibility/context; no donor swap and no CUB backwash rate"
echo "recall rule=both species have >=3 positive and >=3 negative images; no all-positive fallback"
echo "RLv2=not assumed; this report decides whether a later label intervention is justified"
echo "training=no"
echo "Slurm=no"
echo "full CUB=deferred because the official full-CUB Koh result is incomplete"

required_files=(
  "$CUB70_ROOT/SUCCESS.json"
  "$CUB70_ROOT/final_test.parquet"
  "$CURATED_DATA/cub70_visibility.parquet"
  "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/selection_indices.json"
  "$FB_ROOT/SUCCESS.json"
  "$FB_ROOT/final_test.parquet"
  "$SWAP_ROOT/SUCCESS.json"
  "$SWAP_ROOT/funnybirds-cbm-s1.csv"
  "$CURATED_DATA/CUB_200_2011/images.txt"
)
for path in "${required_files[@]}"; do
  test -s "$path" || { echo "ERROR: missing required read-only input: $path" >&2; exit 2; }
  echo "input=$path"
done

test -d "$CURATED_DATA/CUB_200_2011/images" || {
  echo "ERROR: missing required image directory: $CURATED_DATA/CUB_200_2011/images" >&2
  exit 2
}
echo "input=$CURATED_DATA/CUB_200_2011/images"

if [[ -s "$CURATED_DATA/CUB_200_2011/attributes/image_attribute_labels.txt" ]]; then
  echo "input=$CURATED_DATA/CUB_200_2011/attributes/image_attribute_labels.txt"
elif [[ -s "$CURATED_DATA/CUB_200_2011/image_attribute_labels.txt" ]]; then
  echo "input=$CURATED_DATA/CUB_200_2011/image_attribute_labels.txt"
else
  echo "ERROR: missing CUB image-level attribute labels under $CURATED_DATA/CUB_200_2011" >&2
  exit 2
fi

if [[ -d "$CURATED_DATA/cub70/masks/AnnotationMasksPerclass" ]]; then
  echo "input=$CURATED_DATA/cub70/masks/AnnotationMasksPerclass"
elif [[ -d "$CURATED_DATA/cub70/masks" ]]; then
  echo "input=$CURATED_DATA/cub70/masks"
else
  echo "ERROR: missing released CUB70 masks under $CURATED_DATA/cub70/masks" >&2
  exit 2
fi

python analysis/canonical_manifest.py verify --manifest "$CUB70_ROOT/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$FB_ROOT/SUCCESS.json"
python analysis/canonical_manifest.py verify --manifest "$SWAP_ROOT/SUCCESS.json"

echo "[1/5] Run the matched-recall synthetic checks"
python analysis/test_matched_recall_proxy.py

echo "[2/5] Rebuild Notebook 05 and compile every generated code cell"
python analysis/test_cub70_report_builder.py
python analysis/build_standard_cbm_reports.py --only 05

echo "[3/5] Execute the report from official artifacts; no training"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/05_cub_cbm.ipynb

echo "[4/5] Export standalone HTML"
jupyter nbconvert --to html notebooks/05_cub_cbm.ipynb

echo "[5/5] Restore and verify figure alternative text"
python analysis/repair_nbconvert_alt_text.py \
  notebooks/05_cub_cbm.ipynb \
  notebooks/05_cub_cbm.html

echo "Executed notebook: $CURATED/notebooks/05_cub_cbm.ipynb"
echo "Rendered report:   $CURATED/notebooks/05_cub_cbm.html"
echo "NEXT: inspect every current figure and replace the cold-review slots with the observed official-Koh results before publication"
