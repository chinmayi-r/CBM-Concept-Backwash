#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1              # GPU node needed for WebGL display stack
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=fb_make_labels
#SBATCH --output=logs/fb_make_labels_%j.out

set -e
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cubvision-gpu

RENDERER_DIR="/scratch/network/cr7998/funnybirds/render"
FB_ROOT="data/FunnyBirds"
RENDERER_PORT=8081

# Stable tmpdir for Puppeteer so OS cleanup doesn't break mid-run Chrome profiles
export TMPDIR="${SLURM_SUBMIT_DIR}/.puppeteer_tmp"
mkdir -p "$TMPDIR"

# Kill any renderer already on this port so we start fresh
fuser -k ${RENDERER_PORT}/tcp 2>/dev/null || true
sleep 2

# Delete all-zeros checkpoint from any previous failed run
python3 - <<'PYCHECK'
import numpy as np, pathlib, sys
p = pathlib.Path('data/FunnyBirds/pixel_counts_train.npy')
if p.exists():
    px = np.load(p)
    if (px.sum(axis=1) > 0).sum() == 0:
        p.unlink()
        print('[labels] Deleted all-zero checkpoint', file=sys.stderr)
PYCHECK

# Start renderer (server.js always binds to port 8081 — no --port flag)
echo "[fb_make_labels] Starting renderer ..."
node "${RENDERER_DIR}/server.js" &
RENDERER_PID=$!

# Wait up to 30 s
RENDERER_UP=0
for i in $(seq 1 30); do
    sleep 1
    if curl -sf "http://localhost:${RENDERER_PORT}/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" > /dev/null 2>&1; then
        echo "[fb_make_labels] Renderer up after ${i}s"
        RENDERER_UP=1
        break
    fi
done

if [ $RENDERER_UP -eq 0 ]; then
    echo "[fb_make_labels] ERROR: renderer did not start. Exiting."
    kill $RENDERER_PID 2>/dev/null || true
    exit 1
fi

python -m scripts.make_image_level_concept_labels \
    --funnybirds_root "${FB_ROOT}" \
    --output_dir      "${FB_ROOT}" \
    --renderer_url    "http://localhost:${RENDERER_PORT}" \
    --threshold       0.05 \
    --workers         4 \
    --checkpoint_every 2000

echo "[fb_make_labels] Done."
kill "${RENDERER_PID}" 2>/dev/null || true
