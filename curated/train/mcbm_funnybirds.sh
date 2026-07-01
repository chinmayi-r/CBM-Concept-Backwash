#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=curated_mcbm_fb
#SBATCH --output=logs/curated_mcbm_fb_%A_%a.out
#SBATCH --array=0-7               # one job per gamma value
#
# Train MCBM on FunnyBirds via curated/patches/run_mcbm_funnybirds.py, which
# registers a FunnyBirds dataset into the official minimal_cbm get_loader()
# dispatcher (upstream has zero FunnyBirds support -- see that file's
# docstring and curated/compat/funnybirds_mcbm_dataset.py).
#
# Same config-template/sed-render mechanism as mcbm_cub.sh: BaseExperiment
# hardcodes config lookup to <minimal_cbm_root>/configs/<name.split('-')[0]>/
# <name>.yaml, so curated/train/configs/funnybirds-mcbm.yaml is rendered to
# external/minimal_cbm/configs/funnybirds/funnybirds-mcbm.yaml before each run.
#
# Gamma sweep mirrors run_fb_mcbm_train_adroit.sh's set, extended per request
# to also cover 0.05/0.2/2.5 (same 8 values as mcbm_cub.sh, for a like-for-like
# comparison across datasets).
set -e    # stop on error (not -u/pipefail: module load/conda activate reference unset vars like $PS1 in a non-interactive batch shell, which -u turns into a hard crash)
set -x

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2025.6
conda activate cubvision-gpu

: "${CURATED_DATA:?set CURATED_DATA}"
: "${FUNNYBIRDS_ROOT:?set FUNNYBIRDS_ROOT to the official FunnyBirds dataset root (has parts.json)}"
export WANDB_MODE=offline WANDB_DISABLED=true CURATED_DATA
SEED="${1:-42}"
MCBM="external/minimal_cbm"
TEMPLATE="train/configs/funnybirds-mcbm.yaml"

GAMMAS=(0.0 0.05 0.1 0.2 0.5 1.0 2.5 5.0)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$MCBM/configs/funnybirds"
sed -e "s#__CURATED_DATA__#${CURATED_DATA}#g" -e "s/__GAMMA__/${GAMMA}/g" \
    -e "s#__FUNNYBIRDS_ROOT__#${FUNNYBIRDS_ROOT}#g" \
  "$TEMPLATE" > "$MCBM/configs/funnybirds/funnybirds-mcbm.yaml"

cd "$MCBM"
echo "### MCBM FunnyBirds  gamma=${GAMMA}  seed=$SEED"
python3 ../../patches/run_mcbm_funnybirds.py funnybirds-mcbm -s "$SEED"
echo "Done. gamma=${GAMMA}"
