#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CURATED_DATA:?set CURATED_DATA to the curated_data directory}"
export PYTHONUNBUFFERED=1

echo "GOAL: determine where MCBM loses exact-value response and whether swap-induced off-target h changes affect the frozen species head"
echo "INPUTS: accepted seed-1 gamma sweep, accepted fixed swaps, accepted counterfactual-h replay caches"
echo "WORK: reuse counterfactual caches, infer 250 originals per gamma, calibrate h by ordinary labels, compute direct winners, and run one frozen-head hybrid intervention"
echo "TRAINING: no"
echo "SLURM: no"

python analysis/test_mcbm_swap_pathway_report.py
python analysis/mcbm_swap_pathway_report.py \
  --output "$CURATED_DATA/mcbm_swap_pathway_v2"

echo "Completed pathway report: $CURATED_DATA/mcbm_swap_pathway_v2"
