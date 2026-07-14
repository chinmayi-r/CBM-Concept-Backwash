#!/usr/bin/env bash
# Run the deletion grounding test on EVERY trained FunnyBirds CBM/MCBM checkpoint,
# then collect BACKWASH-vs-gamma. Idempotent: skips models already scored.
# (vanilla has no concept head -> nothing to be backwashed -> skipped.)
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # curated/analysis
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"
OUT="$CURATED_DATA/grounding"; mkdir -p "$OUT"
SPOUT="$CURATED_DATA/species_probe"; mkdir -p "$SPOUT"
LIMIT="${LIMIT:-0}"                                        # 0 = all test images

shopt -s nullglob
for md in "$MCBM"/results/funnybirds-*/*/models; do
  seed="$(basename "$(dirname "$md")")"
  config="$(basename "$(dirname "$(dirname "$md")")")"
  case "$config" in *vanilla*) continue;; esac            # no concepts -> skip
  ckpts=("$md"/epoch_*.pt); [ ${#ckpts[@]} -gt 0 ] || continue
  out="$OUT/${config}-s${seed}.parquet"
  # NOTE: skip the grounding STEP if its parquet exists, but do NOT `continue` the whole
  # iteration — that used to skip the species probe below for every already-scored model,
  # which is why only the (retrained) g3 probe existed. Guard each step independently.
  if [ -f "$out" ]; then
    echo "skip grounding (done): $out"
  else
    echo ">>> grounding  $config  seed=$seed"
    ( cd "$CURATED" && python3 analysis/grounding_deletion.py \
        --config "$config" --seed "$seed" \
        --funnybirds-root "$CURATED_DATA/FunnyBirds" \
        --pkls "$CURATED_DATA/funnybirds_processed" \
        --out "$out" --limit "$LIMIT" ) || echo "  FAILED: $config s$seed (continuing)"
  fi

  # species-identity probe on the same checkpoint (renderer-free mechanism metric)
  spout="$SPOUT/${config}-s${seed}.json"
  if [ -f "$spout" ]; then echo "skip (done): $spout"; else
    echo ">>> species-probe  $config  seed=$seed"
    ( cd "$CURATED" && python3 analysis/species_probe.py \
        --config "$config" --seed "$seed" \
        --funnybirds-root "$CURATED_DATA/FunnyBirds" \
        --pkls "$CURATED_DATA/funnybirds_processed" \
        --out "$spout" --limit "$LIMIT" ) || echo "  FAILED (probe): $config s$seed (continuing)"
  fi
done

echo "### collecting BACKWASH vs gamma ###"
python3 "$CURATED/analysis/collect_backwash.py" --grounding "$OUT" \
  --out "$CURATED_DATA/backwash_vs_gamma"
