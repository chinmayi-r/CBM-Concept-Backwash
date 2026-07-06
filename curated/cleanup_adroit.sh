#!/usr/bin/env bash
# Reclaim disk on adroit by removing OLD vibe-coded pipeline artifacts that the
# curated restart regenerates from official code. RAW datasets are kept and
# reused (see the KEEP list). Old figures/notebooks are intentionally NOT touched.
#
# Run from the PROJECT ROOT (the dir that contains curated/), e.g.
#   /scratch/network/cr7998/cv_emergence_project
#
# Dry-run by default (prints what it WOULD delete). To actually delete:
#   APPLY=1 bash curated/cleanup_adroit.sh
set -euo pipefail

ROOT="$(pwd)"
[ -d "$ROOT/curated" ] || { echo "run from the project root (dir containing curated/)"; exit 1; }
APPLY="${APPLY:-0}"

# Things we DELETE (regenerable old artifacts). Sizes are approximate.
DELETE_DIRS=(
  features                 # ~65G  old extracted feature tensors
  checkpoints              # ~92M  old resnet ckpt
  checkpoints_cbm          # ~272M old CBM ckpts
  checkpoints_funnybirds   # ~1.5G old MCBM ckpts (WRONG gamma scaling)
  data/FunnyBirds_code     # ~79M  superseded by curated/external/funnybirds
)
DELETE_GLOBS=(
  'data/FunnyBirds.zip'    # already extracted to data/FunnyBirds
  'data/FunnyBirds.zip.1'  # partial re-download junk
  'fb_mcbm_z_ordering_*.csv'  # old root-level analysis dumps
)

# Things we KEEP and REUSE (never delete). Raw datasets live only on disk
# (gitignored), so they are NOT recoverable from GitHub.
KEEP=(
  data/FunnyBirds          # raw FunnyBirds renders  -> reuse via CURATED_DATA
  data/CUB_200_2011        # raw CUB images          -> reuse via CURATED_DATA
  curated_data             # already-produced curated artifacts
  funnybird_notebooks notebooks figures   # old results (user asked to keep)
)

echo "### cleanup (APPLY=$APPLY)  root=$ROOT"
echo "--- KEEP (reused / preserved) ---"
for k in "${KEEP[@]}"; do [ -e "$ROOT/$k" ] && printf '  keep  %-24s %s\n' "$k" "$(du -sh "$ROOT/$k" 2>/dev/null | cut -f1)"; done

echo "--- DELETE ---"
rm_path() {
  local p="$1"
  [ -e "$p" ] || return 0
  local sz; sz="$(du -sh "$p" 2>/dev/null | cut -f1)"
  if [ "$APPLY" = "1" ]; then rm -rf "$p"; printf '  DELETED %-24s %s\n' "$(basename "$p")" "$sz"
  else printf '  would rm %-24s %s\n' "${p#$ROOT/}" "$sz"; fi
}
for d in "${DELETE_DIRS[@]}"; do rm_path "$ROOT/$d"; done
for g in "${DELETE_GLOBS[@]}"; do
  # shellcheck disable=SC2086
  for f in $ROOT/$g; do rm_path "$f"; done
done

if [ "$APPLY" != "1" ]; then
  echo ">>> dry-run only. Re-run with: APPLY=1 bash curated/cleanup_adroit.sh"
fi
