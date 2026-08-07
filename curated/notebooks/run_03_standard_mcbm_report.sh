#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CURATED_DATA:?set CURATED_DATA to the curated_data directory}"

cat >&2 <<'EOF'
INCOMPLETE: notebook 03 is intentionally disabled until explicitly seeded MCBM
checkpoints and their fixed-render evaluations replace the legacy unseeded runs.
The validated RGB render cache itself remains reusable.
EOF
exit 2

# Re-check the accepted cache at report time. A directory name is not evidence:
# the validator requires identical render IDs and byte hashes across models,
# render diversity, changed RGB pixels, and usable target part maps.
python analysis/validate_fixed_swaps.py \
  --out "$CURATED_DATA/swap_fixed_v2_attempt2"

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
