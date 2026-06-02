#!/bin/bash
#SBATCH -p gpu                    # Must be GPU partition — CPU nodes lack display for WebGL
#SBATCH --gres=gpu:1              # Reserves a GPU node (renderer needs display stack)
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=fb_make_labels
#SBATCH --output=logs/fb_make_labels_%j.out

# Generate image-level concept labels for FunnyBirds training images.
#
# What this does:
#   1. Starts the FunnyBirds 3D renderer (node server.js)
#   2. Renders a part-map segmentation for each of the 50,000 training images
#   3. Counts part pixels per image; compares to species-level medians
#   4. Relabels concept=0 where pixel_count < 5% of species median
#   5. Saves data/FunnyBirds/concept_labels_image_level.npy  (50000, 26)
#
# Output files (all in data/FunnyBirds/):
#   pixel_counts_train.npy          - raw pixel counts, used as checkpoint
#   concept_labels_image_level.npy  - drop-in replacement concept labels
#   concept_label_stats.json        - per-part relabel fractions
#
# Resubmitting is safe: pixel_counts_train.npy is a checkpoint; already-
# rendered images are skipped on restart.

set -e
set -x

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

module load anaconda3/2025.6
conda activate cubvision-gpu

RENDERER_DIR="/scratch/network/cr7998/funnybirds/render"
FB_ROOT="data/FunnyBirds"
RENDERER_PORT=8082   # use 8082 to avoid conflict with Jupyter session on 8081

# Node.js: use full path since it may not be on PATH for SLURM jobs.
# Find it with: which node   (run on adroit-vis where it works)
NODE_BIN="$(which node 2>/dev/null || echo /usr/local/bin/node)"
if [ ! -x "$NODE_BIN" ]; then
    # Try loading a module
    module load nodejs 2>/dev/null || module load node 2>/dev/null || true
    NODE_BIN="$(which node)"
fi
echo "[fb_make_labels] node: $NODE_BIN  ($(${NODE_BIN} --version))"

# Stable tmpdir for Puppeteer — /tmp gets cleaned on long jobs
export TMPDIR="${SLURM_SUBMIT_DIR}/.puppeteer_tmp"
mkdir -p "$TMPDIR"

# Delete any all-zeros checkpoint from previous failed runs
if python3 -c "
import numpy as np, sys, pathlib
p = pathlib.Path('${FB_ROOT}/pixel_counts_train.npy')
if p.exists():
    px = np.load(p)
    if (px.sum(axis=1) > 0).sum() == 0:
        p.unlink()
        sys.stderr.write('Deleted all-zero checkpoint\n')
" 2>&1; then true; fi

# ── Start renderer ──────────────────────────────────────────────────────────
echo "[fb_make_labels] Starting FunnyBirds renderer on port ${RENDERER_PORT} ..."
"${NODE_BIN}" "${RENDERER_DIR}/server.js" --port "${RENDERER_PORT}" &
RENDERER_PID=$!
echo "[fb_make_labels] Renderer PID=${RENDERER_PID}"

# Wait for renderer to be ready (up to 30 s)
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${RENDERER_PORT}/render?render_mode=default&beak_model=beak01.glb" > /dev/null 2>&1; then
        echo "[fb_make_labels] Renderer is up after ${i}s"
        break
    fi
    sleep 1
done

# ── Run label generation ─────────────────────────────────────────────────────
python -m scripts.make_image_level_concept_labels \
    --funnybirds_root "${FB_ROOT}" \
    --output_dir      "${FB_ROOT}" \
    --renderer_url    "http://localhost:${RENDERER_PORT}" \
    --threshold       0.05 \
    --workers         2 \
    --checkpoint_every 2000

echo "[fb_make_labels] Done."

# ── Cleanup renderer ─────────────────────────────────────────────────────────
kill "${RENDERER_PID}" 2>/dev/null || true
