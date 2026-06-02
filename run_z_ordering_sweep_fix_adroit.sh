#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=fb_sweep_fix
#SBATCH --output=logs/fb_sweep_fix_%j.out

# Z-ordering sweep for FIXED MCBM checkpoints (_fix suffix).
# Reads:    checkpoints_funnybirds/mcbm_fb_gamma{G}_fix.pth
#           features/resnet50_mcbm_fb_gamma{G}_fix/
# Outputs:  fb_mcbm_z_ordering_gamma{G}_fix_{part}_v2.csv
#           fb_mcbm_z_ordering_gamma{G}_fix_v2.csv  (combined per gamma)
# Run after both run_fb_mcbm_train_fix_adroit and run_fb_mcbm_extract_fix_adroit complete.

set -e
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cubvision-gpu

RENDERER_DIR="/scratch/network/cr7998/funnybirds/render"

node "${RENDERER_DIR}/server.js" > "/tmp/renderer_${SLURM_JOB_ID}.log" 2>&1 &
RENDERER_PID=$!

RENDERER_UP=0
for i in $(seq 1 30); do
    sleep 1
    curl -sf "http://localhost:8081/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" > /dev/null 2>&1 && RENDERER_UP=1 && break
    echo "  waiting for renderer... ${i}s"
done
if [ $RENDERER_UP -eq 0 ]; then
    echo "[ERROR] Renderer not up. Log:"; cat "/tmp/renderer_${SLURM_JOB_ID}.log"
    kill $RENDERER_PID 2>/dev/null; exit 1
fi

python run_z_ordering_sweep.py \
    --gammas 0.0 0.1 0.5 1.0 5.0 \
    --ckpt_suffix _fix \
    --workers 4 \
    --force

EXIT_CODE=$?
kill $RENDERER_PID 2>/dev/null
echo "Done: $(date)  exit_code=$EXIT_CODE"
exit $EXIT_CODE
