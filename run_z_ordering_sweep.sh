#!/bin/bash
#SBATCH --job-name=mcbm_sweep
#SBATCH --output=logs/mcbm_sweep_%j.out
#SBATCH --error=logs/mcbm_sweep_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p logs

conda activate cubvision-gpu

cd /scratch/network/cr7998/cv_emergence_project

# ── start renderer ────────────────────────────────────────────────────────────
RENDERER_DIR="/scratch/network/cr7998/funnybirds/render"

cd "$RENDERER_DIR"
node server.js > /tmp/renderer_${SLURM_JOB_ID}.log 2>&1 &
RENDERER_PID=$!
cd /scratch/network/cr7998/cv_emergence_project
echo "Renderer PID: $RENDERER_PID  (log: /tmp/renderer_${SLURM_JOB_ID}.log)"

# Poll until renderer responds (up to 30s)
RENDERER_UP=0
for i in $(seq 1 30); do
    sleep 1
    curl -s --max-time 2 \
      "http://localhost:8081/render?render_mode=default&beak_model=beak01.glb&eye_model=eye01.glb&foot_model=foot01.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing01.glb&wing_color=red&camera_distance=300&camera_pitch=0&camera_roll=0&light_distance=300&light_pitch=0&light_roll=0" \
      > /dev/null 2>&1 && RENDERER_UP=1 && break
    echo "  waiting for renderer... ${i}s"
done

if [ $RENDERER_UP -eq 0 ]; then
    echo "[ERROR] Renderer did not start. Node log:"
    cat /tmp/renderer_${SLURM_JOB_ID}.log
    kill $RENDERER_PID 2>/dev/null
    exit 1
fi
echo "Renderer is up."

# ── run sweep ─────────────────────────────────────────────────────────────────
python run_z_ordering_sweep.py --gammas 0.0 0.1 0.5 1.0 5.0 --force
EXIT_CODE=$?

kill $RENDERER_PID 2>/dev/null
echo "Job finished: $(date)  exit_code=$EXIT_CODE"
exit $EXIT_CODE