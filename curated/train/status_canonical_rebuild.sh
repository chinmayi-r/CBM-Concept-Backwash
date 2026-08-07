#!/usr/bin/env bash
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
CANONICAL_ROOT="${CANONICAL_ROOT:-$CURATED_DATA/canonical_20260806_v1}"
TRACK="$CANONICAL_ROOT/submitted_jobs.tsv"
test -s "$TRACK" || { echo "INCOMPLETE: no submission table at $TRACK"; exit 2; }
IDS=$(tail -n +2 "$TRACK" | cut -f1 | paste -sd, -)
echo "===== CURRENT USER QUEUE ====="
squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"
echo "===== CANONICAL ACCOUNTING ====="
sacct -j "$IDS" -X --format=JobID,JobName%30,State,Elapsed,ExitCode,End
echo "===== SUCCESS MANIFEST COUNTS ====="
printf 'successful manifests: '
find "$CANONICAL_ROOT/manifests" -maxdepth 1 -type f -name '*.json' | wc -l
printf 'submitted Slurm jobs (294 stages + final verifier): '
tail -n +2 "$TRACK" | wc -l
python3 "${REPO:-$(git rev-parse --show-toplevel)}/curated/analysis/verify_canonical_completion.py" \
  --root "$CANONICAL_ROOT" || true
echo "===== ERRORS REQUIRING CODE/CONFIG CORRECTION ====="
sacct -j "$IDS" -X -n -P --format=JobID,JobName,State,ExitCode |
  awk -F'|' '$3 ~ /FAILED|OUT_OF_MEMORY|NODE_FAIL|CANCELLED|TIMEOUT/ {print}' || true
