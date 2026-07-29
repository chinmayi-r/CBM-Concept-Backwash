#!/usr/bin/env bash
# One-shot state check: what's running, what finished, what data exists.
# Read-only. Run:  bash status.sh   (from curated/, with CURATED_DATA set)
: "${CURATED_DATA:=/scratch/network/cr7998/cv_emergence_project/curated_data}"
SW="$CURATED_DATA/swap"
SW_FIXED="$CURATED_DATA/swap_fixed_v2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCBM="$HERE/external/minimal_cbm"

echo "CURATED_DATA=$CURATED_DATA"
echo
echo "===== 1. RUNNING / QUEUED NOW ====="
squeue -u "$USER" -o "%.10i %.22j %.3t %.11M %.11L %R" 2>/dev/null || echo "  (squeue unavailable)"

echo
echo "===== 2. FINISHED / FAILED (last 4 days) ====="
sacct -u "$USER" --starttime now-4days -X \
  --format=JobID,JobName%22,State,Elapsed,End 2>/dev/null | \
  grep -vE "extern|batch" || echo "  (sacct unavailable)"

echo
echo "===== 3. SWAP CSVs — which (config, seed) are COMPLETE ====="
echo "     (a seed shows only if its combined -sN.csv exists; per-part files ignored)"
for cfg in funnybirds-cbm \
           funnybirds-mcbm-g0 funnybirds-mcbm-g0p1 funnybirds-mcbm-g0p3 \
           funnybirds-mcbm-g1 funnybirds-mcbm-g3 funnybirds-mcbm-g5 \
           funnybirds-cbm-rlv2 \
           funnybirds-mcbm-rlv2-g0 funnybirds-mcbm-rlv2-g0p1; do
  s=$(ls "$SW/${cfg}"-s*.csv 2>/dev/null | grep -E "/${cfg}-s[0-9]+\.csv$" \
        | grep -oE 's[0-9]+\.csv$' | sed 's/\.csv//' | sort | tr '\n' ' ')
  printf "  %-26s %s\n" "$cfg" "${s:-—none—}"
done

echo
echo "===== 3b. SEMANTICALLY VALIDATED FIXED-RENDER CSVs (v2) ====="
if [ -f "$SW_FIXED/renderer_preflight/renderer_semantic_preflight.json" ]; then
  echo "  semantic preflight artifact: present"
else
  echo "  semantic preflight artifact: —missing—"
fi
for cfg in funnybirds-cbm funnybirds-cbm-rlv2 \
           funnybirds-mcbm-g0 funnybirds-mcbm-g0p1 \
           funnybirds-mcbm-rlv2-g0 funnybirds-mcbm-rlv2-g0p1; do
  if [ -f "$SW_FIXED/${cfg}-s1.csv" ]; then
    printf "  %-30s s1\n" "$cfg"
  else
    printf "  %-30s —missing—\n" "$cfg"
  fi
done
if ls "$SW_FIXED"/*-s1.csv >/dev/null 2>&1; then
  python3 "$HERE/analysis/validate_fixed_swaps.py" --out "$SW_FIXED" \
    2>&1 | tail -n 2
else
  echo "  validation: pending (no combined fixed-render CSVs)"
fi

echo
echo "===== 4. TRAINED MODELS (results dirs + saved epochs) ====="
if [ -d "$MCBM/results" ]; then
  for d in "$MCBM"/results/funnybirds-*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    seeds=$(ls -d "$d"/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
    ep=$(ls "$d"/*/models/epoch_*.pt 2>/dev/null | grep -oE 'epoch_[0-9]+' | sort -u | tr '\n' ' ')
    printf "  %-26s seeds:[%s] epochs:[%s]\n" "$name" "${seeds:-none}" "${ep:-none}"
  done
else echo "  (no results dir at $MCBM/results)"; fi

echo
echo "===== 5. RL relabeled DATA built? ====="
if [ -f "$CURATED_DATA/funnybirds_processed_rl/train.pkl" ]; then
  echo "  YES  $(ls -la "$CURATED_DATA/funnybirds_processed_rl/"*.pkl 2>/dev/null | wc -l) pkl(s)"
else echo "  NO  (funnybirds_processed_rl/train.pkl missing -> RL build not done)"; fi

echo
echo "===== 6. ANALYSIS OUTPUTS ====="
echo "  species_probe json : $(ls "$CURATED_DATA"/species_probe/*.json 2>/dev/null | wc -l)"
echo "  grounding parquet  : $(ls "$CURATED_DATA"/grounding/*.parquet 2>/dev/null | wc -l)"
echo "  backwash_vs_gamma  : $([ -f "$CURATED_DATA/backwash_vs_gamma.csv" ] && echo yes || echo no)"
echo
echo "done."
