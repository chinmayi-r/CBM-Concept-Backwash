#!/usr/bin/env bash
# Submit every missing official-Koh CBM training cell. Safe to rerun: the
# per-stage submitter skips completed manifests and identically named live jobs.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/submit_koh_joint_stage.sh" funnybirds standard "1 2 3"
bash "$HERE/submit_koh_joint_stage.sh" funnybirds rlv2 "1 2 3"
bash "$HERE/submit_koh_joint_stage.sh" cub70 standard "1 2 3"
bash "$HERE/submit_koh_joint_stage.sh" cub standard "1 2 3"

echo "===== ALL MISSING KOH CBM CELLS SUBMITTED OR ALREADY PRESENT ====="
squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"
