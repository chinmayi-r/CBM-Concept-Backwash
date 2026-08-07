#!/usr/bin/env bash
# Compact wake-up table for official-Koh training and fixed-render evaluation.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"

declare -A live
while read -r name state; do
  [ -z "$name" ] || live["$name"]="$state"
done < <(squeue -h -u "$USER" -o "%j %T")

cell_status() {
  local name="$1" success="$2"
  if [ -s "$success" ]; then printf 'DONE'; return; fi
  case "${live[$name]:-}" in
    RUNNING|COMPLETING) printf 'RUNNING' ;;
    PENDING|CONFIGURING) printf 'PENDING' ;;
    *)
      local state
      state=$(sacct -n -X -S 2026-08-06 -u "$USER" --name "$name" \
        --format=State 2>/dev/null | awk 'NF {s=$1} END {print s}')
      case "$state" in
        COMPLETED) printf 'INCOMPLETE_OUTPUT' ;;
        FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*) printf 'ERROR:%s' "$state" ;;
        *) printf 'NOT_SUBMITTED' ;;
      esac
      ;;
  esac
}

printf '%-12s %-10s %-6s %-18s\n' DATASET LABELS SEED STATUS
printf '%-12s %-10s %-6s %-18s\n' '------------' '----------' '------' '------------------'
for dataset in funnybirds cub70 cub; do
  labels=standard
  for seed in 1 2 3; do
    name="koh_${dataset}_${labels}_s${seed}"
    success="$CURATED_DATA/koh_joint_v1/$dataset/$labels/seed$seed/SUCCESS.json"
    printf '%-12s %-10s %-6s %-18s\n' "$dataset" "$labels" "$seed" \
      "$(cell_status "$name" "$success")"
  done
  if [ "$dataset" = funnybirds ]; then
    labels=rlv2
    for seed in 1 2 3; do
      name="koh_${dataset}_${labels}_s${seed}"
      success="$CURATED_DATA/koh_joint_v1/$dataset/$labels/seed$seed/SUCCESS.json"
      printf '%-12s %-10s %-6s %-18s\n' "$dataset" "$labels" "$seed" \
        "$(cell_status "$name" "$success")"
    done
  fi
done

printf '\n%-28s %-18s\n' DOWNSTREAM STATUS
printf '%-28s %-18s\n' funnybird_fixed_swaps \
  "$(cell_status koh_fb_swaps "$CURATED_DATA/swap_koh_joint_v1/SUCCESS.json")"

printf '\n===== LIVE QUEUE =====\n'
squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"

