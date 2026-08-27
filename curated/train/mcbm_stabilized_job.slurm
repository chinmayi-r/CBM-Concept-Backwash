#!/usr/bin/env bash
# Dedicated full-GPU launcher for the declared CUB70 stabilized high-gamma bridge.
# FP32 at 299 pixels and batch 64 may exceed the old 20-GB MIG allocation.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=mcbm_stabilized_%x_%j.out

set -euo pipefail

module load anaconda3/2025.6
conda activate cubvision-gpu

cd "${REPO:?set REPO}"
bash curated/train/mcbm_seeded_stage.sh
