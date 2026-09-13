#!/usr/bin/env bash
#SBATCH --job-name=fb_4cell_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail

: "${REPO:?set REPO}"
: "${CURATED_DATA:?set CURATED_DATA}"

CONDA_ENV="${CONDA_ENV:-cubvision-gpu}"
RENDERER_DIR="${RENDERER_DIR:-/scratch/network/cr7998/funnybirds/render}"
RENDERER_URL="${RENDERER_URL:-http://localhost:8081}"
PARTS_KEY="${PARTS_KEY:-tail-wing}"
PARTS="${PARTS_KEY//-/ }"
MAX_IMAGES_PER_PART="${MAX_IMAGES_PER_PART:-25}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}}"
OUT="${OUT:-$CURATED_DATA/funnybird_four_condition_v1/$RUN_ID}"
CHECKPOINT="${CHECKPOINT:-$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/final_model_1.pth}"

echo "GOAL: separate named-part evidence from remaining-bird context evidence"
echo "WORK: four renderer conditions and frozen Koh Joint inference; no training"
echo "MODEL: accepted seed-1 ResNet-50 Koh-architecture Joint CBM, Standard labels"
echo "PARTS: $PARTS"
echo "ROWS: at most $MAX_IMAGES_PER_PART balanced test images per part"
echo "CONDITIONS: symmetric pixel factorial; the same target pixels are pasted into both present cells"
echo "CONTEXT BOUNDARY: only the four other named part meshes are removed; base body/background remain"
echo "OUTPUT: $OUT"
echo "STAGES: synthetic test -> accepted-manifest check -> renderer semantic gate -> pilot render/infer -> visual-calibration artifact"

cd "$REPO"
module load anaconda3/2025.6 2>/dev/null || true
set +u
conda activate "$CONDA_ENV"
set -u
export WANDB_MODE=offline WANDB_DISABLED=true
ulimit -c 0 || true

python curated/analysis/test_funnybird_four_condition.py
test -s "$CHECKPOINT"
test -s "$CURATED_DATA/FunnyBirds/parts.json"
test -s "$CURATED_DATA/FunnyBirds/dataset_test.json"
test -f "$RENDERER_DIR/server.js"
test ! -e "$OUT" || { echo "ERROR: output already exists: $OUT"; exit 2; }
python curated/analysis/canonical_manifest.py verify \
  --manifest "$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds/standard/seed1/SUCCESS.json"

if curl -s --max-time 2 "$RENDERER_URL/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" >/dev/null; then
  echo "ERROR: a renderer already responds at $RENDERER_URL; refusing to risk using a stale process"
  exit 2
fi

RENDERER_LOG="/tmp/fb_4cell_renderer_${SLURM_JOB_ID:-manual}.log"
( cd "$RENDERER_DIR" && exec node server.js > "$RENDERER_LOG" 2>&1 ) &
RPID=$!
cleanup() { kill "$RPID" 2>/dev/null || true; }
trap cleanup EXIT

UP=0
for attempt in $(seq 1 30); do
  if ! kill -0 "$RPID" 2>/dev/null; then
    echo "ERROR: the renderer process exited before becoming ready"
    cat "$RENDERER_LOG"
    exit 2
  fi
  if curl -s --max-time 2 "$RENDERER_URL/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" >/dev/null; then
    UP=1
    break
  fi
  sleep 1
done
if [ "$UP" -ne 1 ]; then
  echo "ERROR: renderer did not start"
  cat "$RENDERER_LOG"
  exit 2
fi

python curated/analysis/z_ordering_swap.py \
  --config-prefix funnybirds-cbm \
  --gammas 0 \
  --seeds 1 \
  --funnybirds-root "$CURATED_DATA/FunnyBirds" \
  --renderer-url "$RENDERER_URL" \
  --out "$OUT/renderer_gate" \
  --workers 1 \
  --max-pairs 1 \
  --max-imgs 1 \
  --preflight-only \
  --no-v2
kill -0 "$RPID" 2>/dev/null || {
  echo "ERROR: renderer exited after semantic preflight"
  cat "$RENDERER_LOG"
  exit 2
}

set +e
python curated/analysis/funnybird_four_condition.py \
  --funnybirds-root "$CURATED_DATA/FunnyBirds" \
  --checkpoint "$CHECKPOINT" \
  --renderer-url "$RENDERER_URL" \
  --out "$OUT/analysis" \
  --parts $PARTS \
  --max-images-per-part "$MAX_IMAGES_PER_PART"
analysis_rc=$?
set -e

test -s "$OUT/analysis/PILOT_STATUS.json"
python - "$OUT/analysis/PILOT_STATUS.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("PILOT STATUS:", data["status"])
print("scientific_result:", data["scientific_result"])
print("eligible_rows:", data["eligible_rows"], "/", data["selected_rows"])
for item in data["mechanical_gate_failures"]:
    print("  gate:", item)
PY
if [ "$analysis_rc" -ne 0 ]; then
  echo "METHOD NOT CALIBRATED: artifacts were preserved at $OUT for diagnosis"
  exit "$analysis_rc"
fi
echo "ACCEPTED FOR VISUAL-CALIBRATION REVIEW ONLY: $OUT"
echo "No scientific four-condition conclusion is accepted until the galleries are reviewed."
