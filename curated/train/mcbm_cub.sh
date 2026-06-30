#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=curated_mcbm_cub
#SBATCH --output=logs/curated_mcbm_cub_%A_%a.out
#SBATCH --array=0-7               # one job per gamma value
#
# Train MCBM on CUB-200 via the official minimal_cbm bin/train.py.
# WANDB_MODE=offline neutralizes the hardcoded wandb key (patches/README #2).
#
# BaseExperiment.__init__ (src/experiments/base.py) hardcodes config lookup to
# <minimal_cbm_root>/configs/<name.split('-')[0]>/<name>.yaml -- it does not
# take a path. curated/train/configs/cub-mcbm.yaml is therefore a TEMPLATE
# (read_config() is plain yaml.load, no env-var interpolation): this script
# sed-renders __CURATED_DATA__/__GAMMA__ into a real file at
# external/minimal_cbm/configs/cub/cub-mcbm.yaml before each gamma's run, then
# invokes bin/train.py with the bare config name "cub-mcbm".
#
# Gamma sweep mirrors run_mcbm_train_adroit.sh's set, extended per request to
# also cover 0.05/0.2/2.5.
set -euo pipefail
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate mcbm

: "${CURATED_DATA:?set CURATED_DATA}"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA
SEED="${1:-42}"
MCBM="external/minimal_cbm"
TEMPLATE="train/configs/cub-mcbm.yaml"

GAMMAS=(0.0 0.05 0.1 0.2 0.5 1.0 2.5 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$MCBM/configs/cub"
sed -e "s#__CURATED_DATA__#${CURATED_DATA}#g" -e "s/__GAMMA__/${GAMMA}/g" \
  "$TEMPLATE" > "$MCBM/configs/cub/cub-mcbm.yaml"

cd "$MCBM"
echo "### MCBM CUB-200  gamma=${GAMMA}  seed=$SEED"
python3 bin/train.py cub-mcbm -s "$SEED"
echo "Done. gamma=${GAMMA}"
