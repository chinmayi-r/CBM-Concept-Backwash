#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CURATED_DATA:?set CURATED_DATA to the curated_data directory}"

python analysis/build_03_standard_mcbm_report.py

# This runner consumes completed checkpoints and validated fixed renders only.
# It does not submit, release, or cancel Slurm jobs.
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/03_funnybirds_mcbm.ipynb

jupyter nbconvert --to html notebooks/03_funnybirds_mcbm.ipynb
python analysis/repair_nbconvert_alt_text.py \
  notebooks/03_funnybirds_mcbm.ipynb \
  notebooks/03_funnybirds_mcbm.html

echo "Executed standard MCBM report: $ROOT/notebooks/03_funnybirds_mcbm.html"
