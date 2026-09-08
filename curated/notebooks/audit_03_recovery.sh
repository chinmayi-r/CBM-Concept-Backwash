#!/usr/bin/env bash
# Existing evidence only. Each rejected directory is identified by the saved log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CURATED_DATA:?Set CURATED_DATA to the existing curated_data directory}"
echo "GOAL: collect all MCBM recovery evidence before another render or GPU run"
echo "WORK: read existing files; no training, inference, patching or acceptance changes"
python "$ROOT/analysis/audit_mcbm_recovery.py" \
  --rejected 3=bcfc08443a298141bc3495434b48dc03f7dd921d25c33c18bdaa71d80c8d3288 \
  --rejected 5=43a15ce4e56803420271288477239fd3d70d358554249abda6f00b715a0359a0
