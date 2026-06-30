#!/usr/bin/env bash
# One-time setup for the curated pipeline. Run from curated/ on a machine that
# already did `git submodule update --init --recursive`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> Ensuring submodules are populated"
# NOTE: .gitmodules is committed, but the gitlink commits could NOT be created in
# the offline authoring environment. So `git submodule update --init` may find no
# pinned commit. If so, we add the submodules now (one-time, needs network), which
# both pins and populates them. Re-running is a no-op once present.
git submodule sync >/dev/null 2>&1 || true
git submodule update --init --recursive 2>/dev/null || true
for sm in external/ConceptBottleneck external/minimal_cbm; do
  if [ ! -e "$sm/.git" ] && [ ! -f "$sm/.git" ]; then
    case "$sm" in
      *ConceptBottleneck) url=https://github.com/yewsiang/ConceptBottleneck.git ;;
      *minimal_cbm)       url=https://github.com/antonioalmudevar/minimal_cbm.git ;;
    esac
    echo "  $sm empty -> git submodule add --force $url curated/$sm"
    ( cd "$(git rev-parse --show-toplevel)" && git submodule add --force "$url" "curated/$sm" )
  fi
  if [ ! -e "$sm/.git" ] && [ ! -f "$sm/.git" ]; then
    echo "ERROR: could not populate $sm (no network?)." >&2
    exit 1
  fi
done
echo "  After this succeeds once, commit the new gitlinks so future clones just need update --init."

echo "==> Recording commit SHAs for the paper -> external/COMMITS.txt"
{
  echo "Recorded $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ConceptBottleneck $(git -C external/ConceptBottleneck rev-parse HEAD)"
  echo "minimal_cbm       $(git -C external/minimal_cbm rev-parse HEAD)"
} | tee external/COMMITS.txt

echo "==> Building conda environments (CBM and MCBM have incompatible stacks)"
if command -v conda >/dev/null 2>&1; then
  conda env create -f environment-cbm.yml  || echo "  (env 'cbm' may already exist)"
  conda env create -f environment-mcbm.yml || echo "  (env 'mcbm' may already exist)"
else
  echo "  conda not found; create the two envs manually from environment-*.yml" >&2
fi

echo "==> Applying documented compatibility notes"
echo "  Compatibility is handled in curated/compat (import-time shims) and"
echo "  documented in curated/patches/. No files inside external/ are modified."

echo "==> Import sanity checks (run inside each env)"
cat <<'EOF'
  conda run -n cbm  python -c "import sys; sys.path.insert(0,'external/ConceptBottleneck'); import CUB; print('CBM import OK')"
  conda run -n mcbm python -c "import sys; sys.path.insert(0,'external/minimal_cbm'); import src; print('MCBM import OK')"
EOF
echo "==> Done. Next: data/README.md, then train/, then notebooks/."
