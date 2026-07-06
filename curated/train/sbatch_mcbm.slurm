#!/usr/bin/env bash
#SBATCH --job-name=mcbm_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1        # a 20GB MIG slice is plenty for resnet18 MCBM;
                                    # override for full A100:  sbatch --gres=gpu:1 ...
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out          # <jobname>_<jobid>.out in the submit dir
#
# Submit the OFFICIAL-trainer MCBM gamma sweep as a batch job (runs even if you
# log out; better than holding an interactive salloc). SUBMIT FROM curated/ :
#
#   cd /scratch/network/cr7998/cv_emergence_project/curated
#
#   # fast smoke (one point, few epochs) -- pair with a short --time:
#   GAMMAS="30" SEEDS="1" MCBM_MAX_EPOCHS=5 \
#     sbatch --time=00:20:00 train/sbatch_mcbm.slurm funnybirds
#
#   # full grid (long; consider a job array, see note at bottom):
#   GAMMAS="0 0.1 0.3 1 3 10 30" SEEDS="1 2 3" \
#     sbatch --time=24:00:00 train/sbatch_mcbm.slurm funnybirds
#
# Env knobs (export before sbatch; SLURM forwards them by default):
#   GAMMAS, SEEDS, MCBM_MAX_EPOCHS  -> passed through to the sweep / run_mcbm.py
#   CONDA_ENV   (default cubvision-gpu)
#   CURATED_DATA (must already be set/exported)
set -euo pipefail

DATASET="${1:-funnybirds}"
CONDA_ENV="${CONDA_ENV:-cubvision-gpu}"
: "${CURATED_DATA:?export CURATED_DATA before sbatch}"

# submit from curated/ ; fall back to this script's dir/..
CURATED="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -d "$CURATED/train" ] || CURATED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}  dataset=$DATASET"
echo "GAMMAS=[${GAMMAS:-default}]  SEEDS=[${SEEDS:-default}]  MCBM_MAX_EPOCHS=${MCBM_MAX_EPOCHS:-full}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# activate conda non-interactively
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$CURATED"
export WANDB_MODE=offline WANDB_DISABLED=true
bash train/mcbm_gamma_sweep.sh "$DATASET"
echo "sbatch payload done."

# --- Scaling note -------------------------------------------------------------
# The sweep above runs (gamma x seed) SERIALLY in one job. For the full grid,
# parallelize with a job array instead: submit one array task per (gamma,seed)
# by wrapping a single GAMMAS/SEEDS pair per task, e.g. build a task list and use
#   #SBATCH --array=0-20            # 7 gammas x 3 seeds - 1
# then index GAMMAS/SEEDS by $SLURM_ARRAY_TASK_ID. Ask and I'll generate it.
