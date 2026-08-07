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

# Queue one downstream swap job after every currently unfinished FunnyBird
# training job terminates. afterany avoids a permanent DependencyNeverSatisfied
# job; the swap stage itself requires all six SUCCESS manifests before running.
swap_name=koh_fb_swaps
swap_success="$CURATED_DATA/swap_koh_joint_v1/SUCCESS.json"
if [ -s "$swap_success" ]; then
  echo "FunnyBird fixed swaps ALREADY COMPLETE: $swap_success"
elif squeue -h -u "$USER" -n "$swap_name" | grep -q .; then
  echo "FunnyBird fixed swaps ALREADY QUEUED: $swap_name"
else
  dependencies=()
  for labels in standard rlv2; do
    for seed in 1 2 3; do
      success="$CURATED_DATA/koh_joint_v1/funnybirds/$labels/seed$seed/SUCCESS.json"
      [ -s "$success" ] && continue
      name="koh_funnybirds_${labels}_s${seed}"
      jid=$(squeue -h -u "$USER" -n "$name" -o "%A" | head -n 1)
      [ -n "$jid" ] || { echo "ERROR: no completion or live job for $name" >&2; exit 2; }
      dependencies+=("$jid")
    done
  done
  dep_args=()
  if [ "${#dependencies[@]}" -gt 0 ]; then
    joined=$(IFS=:; echo "${dependencies[*]}")
    dep_args=(--dependency="afterany:$joined")
  fi
  swap_jid=$(sbatch --parsable "${dep_args[@]}" --job-name="$swap_name" \
    --export="ALL,REPO=$(git -C "$HERE/../.." rev-parse --show-toplevel),CURATED_DATA=$CURATED_DATA" \
    "$HERE/koh_funnybird_swaps_job.slurm")
  echo "FunnyBird fixed swaps job=$swap_jid dependencies=${dependencies[*]:-none}"
fi

echo "===== ALL MISSING KOH CBM CELLS SUBMITTED OR ALREADY PRESENT ====="
squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"
