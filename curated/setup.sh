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

echo "==> Building conda environments (CBM and MCBM have incompatible stacks)"
if command -v conda >/dev/null 2>&1; then
  conda env create -f environment-cbm.yml  || echo "  (env 'cbm' may already exist)"
  conda env create -f environment-mcbm.yml || echo "  (env 'mcbm' may already exist)"
else
  echo "  conda not found; create the two envs manually from environment-*.yml" >&2
fi

echo "==> Applying curated patches to submodules (idempotent)"
# Each patches/<submodule>.patch is a small, tracked, citable edit applied on top
# of the pinned SHA (submodule stays clean-at-SHA + patch). Named by submodule.
for p in patches/*.patch; do
  [ -f "$p" ] || continue
  sub="external/$(basename "$p" .patch)"
  [ -d "$sub" ] || { echo "  skip $p (no $sub)"; continue; }
  abs="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
  # Reset tracked files to the pinned SHA (drops any prior/older copy of this
  # patch; leaves untracked generated configs/results alone), then apply fresh.
  # Idempotent AND robust to the patch content changing between pulls.
  git -C "$sub" checkout -- . 2>/dev/null || true
  if git -C "$sub" apply "$abs" 2>/dev/null; then
    echo "  applied: $p -> $sub"
  else
    echo "  WARN: could not apply $p to $sub (check manually)" >&2
  fi
done

echo "==> Import sanity checks (run inside each env)"
cat <<'EOF'
  conda run -n cbm  python -c "import sys; sys.path.insert(0,'external/ConceptBottleneck'); import CUB; print('CBM import OK')"
  conda run -n mcbm python -c "import sys; sys.path.insert(0,'external/minimal_cbm'); import src; print('MCBM import OK')"
EOF
echo "==> Done. Next: data/README.md, then train/, then notebooks/."
