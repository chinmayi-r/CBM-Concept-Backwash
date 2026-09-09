#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CURATED_DATA:?set CURATED_DATA to the curated_data directory}"
export PYTHONUNBUFFERED=1

echo "GOAL: determine whether MCBM loses the controlled-swap response in image->h or h->q(h)=z"
echo "INPUTS: accepted seed-1 gamma sweep, accepted fixed swaps, accepted counterfactual-h replay caches"
echo "WORK: infer only 250 unique original images per gamma, then compare matched h and z changes"
echo "TRAINING: no"
echo "SLURM: no"

python analysis/test_mcbm_swap_pathway_report.py
python analysis/mcbm_swap_pathway_report.py \
  --output "$CURATED_DATA/mcbm_swap_pathway_v1"

echo "Completed pathway report: $CURATED_DATA/mcbm_swap_pathway_v1"
