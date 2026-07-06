#!/usr/bin/env bash
# Comprehensive MCBM gamma sweep on the OFFICIAL minimal_cbm trainer.
#
# The scientific spine of the curated restart: identical official code trained
# from scratch across a wide gamma grid x seeds, so every downstream result
# (recall gap, causal swap, grounding) plots AGAINST gamma. The old hand-written
# MCBM used a single, mis-scaled gamma; this fixes both.
#
# gamma is the minimality/bottleneck strength. Effective z-force = gamma * 0.2
# (the 0.2 is baked into minimal_cbm get_loss_z; verified in src/models/mcbm.py).
# gamma=0 = no minimality (control).
#
# HOW IT WORKS (verified against minimal_cbm src reading order):
#   * config is loaded BY BASENAME from  <mcbm>/configs/<prefix>/<basename>.yaml
#     where prefix = basename.split('-')[0]  (base.py). So we write generated
#     configs into external/minimal_cbm/configs/<DATASET>/ (gitignored, generated;
#     no tracked submodule file is touched).
#   * results are written to  <mcbm>/results/<basename>/<seed>/  -> distinct gamma
#     => distinct basename => distinct results dir. NO collision across the grid.
#   * we substitute ONLY the __GAMMA__ token (model.gamma), never scheduler.gamma.
#
# Usage:
#   bash curated/train/mcbm_gamma_sweep.sh funnybirds
#   GAMMAS="0 0.1 0.3 1 3 10 30" SEEDS="1 2 3" bash curated/train/mcbm_gamma_sweep.sh funnybirds
#
# Single smoke test (recommended first):
#   GAMMAS="30" SEEDS="1" bash curated/train/mcbm_gamma_sweep.sh funnybirds
set -euo pipefail
: "${CURATED_DATA:?set CURATED_DATA}"

DATASET="${1:?usage: mcbm_gamma_sweep.sh <funnybirds|cub|cub70>}"
GAMMAS="${GAMMAS:-0 0.1 0.3 1 3 10 30}"   # wide, log-spaced; includes 0 control
SEEDS="${SEEDS:-1 2 3}"                    # >=3 for error bars on 'X vs gamma'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # curated/train
CURATED="$(cd "$HERE/.." && pwd)"
MCBM="$CURATED/external/minimal_cbm"
TEMPLATE="$HERE/configs/${DATASET}-mcbm.yaml"
[ -f "$TEMPLATE" ] || { echo "no template: $TEMPLATE" >&2; exit 1; }

# fabricated-schema guard: refuse to run a template that was never migrated to
# the real minimal_cbm schema (would crash inside read_config / get_model).
if grep -qE '(\bname:|manifest_dir:|n_classes:|\$\{oc\.env)' "$TEMPLATE"; then
  echo "ERROR: $TEMPLATE still uses the OLD fabricated schema (name:/manifest_dir:/oc.env)." >&2
  echo "       Only funnybirds-mcbm.yaml has been migrated so far." >&2
  exit 1
fi

PKLS_DIR="$CURATED_DATA/${DATASET}_processed"
[ -d "$PKLS_DIR" ] || { echo "ERROR: pkls dir missing: $PKLS_DIR (run the data builders first)" >&2; exit 1; }

GEN_DIR="$MCBM/configs/${DATASET}"; mkdir -p "$GEN_DIR"   # configs/<prefix>/

echo "### MCBM gamma sweep  dataset=$DATASET  gammas=[$GAMMAS]  seeds=[$SEEDS]"
echo "    template=$TEMPLATE  pkls=$PKLS_DIR  gen=$GEN_DIR"
for g in $GAMMAS; do
  gtag="${g//./p}"                                          # 0.1 -> 0p1 (dir-safe)
  base="${DATASET}-mcbm-g${gtag}"                           # prefix == $DATASET
  cfg="$GEN_DIR/${base}.yaml"
  sed -e "s|__GAMMA__|${g}|g" -e "s|__PKLS_DIR__|${PKLS_DIR}|g" "$TEMPLATE" > "$cfg"
  # verify exactly the intended substitutions landed and no token survived
  grep -q "__GAMMA__\|__PKLS_DIR__" "$cfg" && { echo "unsubstituted token in $cfg" >&2; exit 1; }
  grep -qE "^[[:space:]]*gamma:[[:space:]]*${g}([^0-9]|$)" "$cfg" || { echo "model.gamma sub failed for $g" >&2; exit 1; }
  grep -qE "^[[:space:]]*gamma:[[:space:]]*0.1([^0-9]|$)" "$cfg" || { echo "scheduler.gamma got clobbered for $g" >&2; exit 1; }
  for s in $SEEDS; do
    echo ">>> gamma=$g seed=$s  ($base)"
    ( cd "$HERE" && python3 run_mcbm.py "$base" -s "$s" )
  done
done
echo "Done. Generated configs in $GEN_DIR ; results in $MCBM/results/<name>/<seed>/."
echo "Next: dump eval tables per run (analysis/io.py), then z-vs-gamma plot."
