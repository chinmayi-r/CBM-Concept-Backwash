#!/usr/bin/env bash
#SBATCH --job-name=fb_swap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#
# Curated port of run_z_ordering_sweep.sh: starts the FunnyBirds Node renderer,
# runs the renderer part-swap / z-ordering sweep on the CURATED minimal_cbm models,
# and writes per-part + combined CSVs to $CURATED_DATA/swap/. Notebook 03 then reads
# those CSVs for the §13-24 analyses (per-gamma margins, fwd/bwd heatmap, IB
# compression-vs-swap, species-encoder signature). Matches the heavy-compute-in-SLURM
# pattern (this is the one analysis that needs the live renderer + GPU).
#
#   export CURATED_DATA=/scratch/network/cr7998/cv_emergence_project/curated_data
#   GAMMAS="0 0.1 0.3 1 3 5" SEEDS="1 2 3" sbatch train/renderer_swap.slurm
set -uo pipefail
: "${CURATED_DATA:?export CURATED_DATA before sbatch}"
CONDA_ENV="${CONDA_ENV:-cubvision-gpu}"
RENDERER_DIR="${RENDERER_DIR:-/scratch/network/cr7998/funnybirds/render}"
RENDERER_URL="${RENDERER_URL:-http://localhost:8081}"
GAMMAS="${GAMMAS:-0 0.1 0.3 1 3 5}"
SEEDS="${SEEDS:-1}"
CURATED="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -d "$CURATED/train" ] || CURATED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
set +u
module load anaconda3/2025.6 2>/dev/null || true
conda activate "$CONDA_ENV"
set -u
export WANDB_MODE=offline WANDB_DISABLED=true

# ── start renderer ────────────────────────────────────────────────────────────
if [ ! -f "$RENDERER_DIR/server.js" ]; then
  echo "[ERROR] renderer not found at $RENDERER_DIR/server.js (set RENDERER_DIR)"; exit 1
fi
( cd "$RENDERER_DIR" && node server.js > "/tmp/renderer_${SLURM_JOB_ID:-manual}.log" 2>&1 ) &
RPID=$!
echo "renderer PID=$RPID"
UP=0
for i in $(seq 1 30); do
  sleep 1
  curl -s --max-time 2 "$RENDERER_URL/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" >/dev/null 2>&1 && UP=1 && break
  echo "  waiting for renderer... ${i}s"
done
[ $UP -eq 1 ] || { echo "[ERROR] renderer did not come up:"; cat "/tmp/renderer_${SLURM_JOB_ID:-manual}.log"; kill $RPID 2>/dev/null; exit 1; }
echo "renderer up."

# ── run the swap sweep on curated models ──────────────────────────────────────
# CONFIG_PREFIX=funnybirds-cbm for the CBM swap (no gamma); funnybirds-mcbm for the sweep.
# Speed knobs: WORKERS (parallel renders), MAX_PAIRS/MAX_IMGS (fewer=faster/coarser),
# NO_V2=1 (skip part-map -> ~2x faster but LOSES the occlusion control).
CONFIG_PREFIX="${CONFIG_PREFIX:-funnybirds-mcbm}"
WORKERS="${WORKERS:-4}"; MAX_PAIRS="${MAX_PAIRS:-100}"; MAX_IMGS="${MAX_IMGS:-5}"
V2FLAG=""; [ "${NO_V2:-0}" = "1" ] && V2FLAG="--no-v2"
( cd "$CURATED" && python3 analysis/z_ordering_swap.py \
    --config-prefix "$CONFIG_PREFIX" --gammas $GAMMAS --seeds $SEEDS \
    --funnybirds-root "$CURATED_DATA/FunnyBirds" \
    --renderer-url "$RENDERER_URL" --out "$CURATED_DATA/swap" \
    --workers "$WORKERS" --max-pairs "$MAX_PAIRS" --max-imgs "$MAX_IMGS" $V2FLAG )
EXIT=$?
kill $RPID 2>/dev/null
echo "done exit=$EXIT"; exit $EXIT
