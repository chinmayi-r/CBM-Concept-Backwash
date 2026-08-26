#!/usr/bin/env bash
# Optional visible coordinator. Each named entry remains independently runnable.
set +e
set +u
set +o pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
ENTRY="$REPO/curated/train/entries"
temporary=$(mktemp "${TMPDIR:-/tmp}/seed1-entry-output.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT

echo "===== CURRENT SEED-1 PLAN ====="
echo "0 reconcile FunnyBird standard (no GPU)"
echo "1 submit FunnyBird RLv2 (independent)"
echo "2 submit CUB70 standard ResNet-50 (independent)"
echo "3 submit full-CUB standard ResNet-50 (independent)"
echo "4 submit FunnyBird swaps (only true consumer; waits for RLv2 if needed)"
echo "No seeds 2/3. No CUB70 MCBM retry. No full-CUB MCBM sweep."

echo "===== RUN ENTRY 0 ====="
bash "$ENTRY/00_reconcile_funnybird_standard_s1.sh"
entry0=$?
echo "ENTRY 0 STATUS=$entry0"

echo "===== RUN ENTRY 1 ====="
bash "$ENTRY/01_submit_funnybird_rlv2_s1.sh" 2>&1 | tee "$temporary"
entry1=${PIPESTATUS[0]}
rl_job=$(sed -n 's/.*\[ENTRY 1 SUBMITTED\] job=\([0-9][0-9]*\).*/\1/p' "$temporary" | tail -n 1)
echo "ENTRY 1 STATUS=$entry1 JOB=${rl_job:-none}"

echo "===== RUN ENTRY 2 (INDEPENDENT OF ENTRY 1) ====="
bash "$ENTRY/02_submit_cub70_standard_s1.sh"
entry2=$?
echo "ENTRY 2 STATUS=$entry2"

echo "===== RUN ENTRY 3 (INDEPENDENT OF ENTRIES 1 AND 2) ====="
bash "$ENTRY/03_submit_full_cub_standard_s1.sh"
entry3=$?
echo "ENTRY 3 STATUS=$entry3"

echo "===== RUN ENTRY 4 ====="
if [ -n "$rl_job" ]; then
  bash "$ENTRY/04_submit_funnybird_swaps_s1.sh" "$rl_job"
else
  bash "$ENTRY/04_submit_funnybird_swaps_s1.sh"
fi
entry4=$?
echo "ENTRY 4 STATUS=$entry4"

echo "===== FINAL ENTRY TABLE ====="
printf '%-8s %-8s %-12s\n' entry status job
printf '%-8s %-8s %-12s\n' 0 "$entry0" none 1 "$entry1" "${rl_job:-none}" \
  2 "$entry2" see-above 3 "$entry3" see-above 4 "$entry4" see-above
echo "===== FINAL LIVE QUEUE ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
