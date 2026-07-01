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
TOP="$(git rev-parse --show-toplevel)"
# iterate every submodule declared in .gitmodules (CBM, MCBM, FunnyBirds x2)
git config -f "$TOP/.gitmodules" --get-regexp '^submodule\..*\.path$' | while read -r key path; do
  name="${key#submodule.}"; name="${name%.path}"
  url="$(git config -f "$TOP/.gitmodules" --get "submodule.${name}.url")"
  if [ ! -e "$TOP/$path/.git" ]; then
    echo "  $path empty -> git submodule add --force $url $path"
    ( cd "$TOP" && git submodule add --force "$url" "$path" )
  fi
  if [ ! -e "$TOP/$path/.git" ]; then
    echo "ERROR: could not populate $path (no network?)." >&2; exit 1
  fi
done
echo "  After this succeeds once, commit the new gitlinks so future clones just need update --init."

echo "==> Recording commit SHAs for the paper -> external/COMMITS.txt"
{
  echo "Recorded $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for d in ConceptBottleneck minimal_cbm funnybirds-framework funnybirds; do
    [ -e "external/$d/.git" ] && printf "%-22s %s\n" "$d" "$(git -C external/$d rev-parse HEAD)"
  done
} | tee external/COMMITS.txt

echo "==> Conda environment"
echo "  train/*.sh all activate 'cubvision-gpu' (the existing real-pipeline env)"
echo "  rather than building separate cbm/mcbm envs -- environment-cbm.yml and"
echo "  environment-mcbm.yml are kept for reference only (a from-scratch setup"
echo "  with no pre-existing env would still need them) but are not created here."
echo "  If cubvision-gpu is missing a dependency either repo needs, add it there"
echo "  rather than introducing a second env; see patches/README.md for known"
echo "  compatibility notes (torchvision Inception weights API, numpy aliases)."

echo "==> Applying documented compatibility notes"
echo "  Compatibility is handled in curated/compat (import-time shims) and"
echo "  documented in curated/patches/. No files inside external/ are modified."

echo "==> Import sanity checks (run inside cubvision-gpu)"
cat <<'EOF'
  conda run -n cubvision-gpu python -c "import sys; sys.path.insert(0,'external/ConceptBottleneck'); import CUB; print('CBM import OK')"
  conda run -n cubvision-gpu python -c "import sys; sys.path.insert(0,'external/minimal_cbm'); import src; print('MCBM import OK')"
EOF
echo "==> Done. Next: data/README.md, then train/, then notebooks/."
