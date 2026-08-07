#!/usr/bin/env bash
# Intended for the persistent Jupyter terminal: build CPU-side data, then submit.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
: "${CUB_ROOT:?export CUB_ROOT}"
SOURCE_REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CANONICAL_ROOT="${CANONICAL_ROOT:-$CURATED_DATA/canonical_20260806_v1}"
FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-$CURATED_DATA/FunnyBirds}"

if [ -z "${CUB_ATTR_FILE:-}" ]; then
  for candidate in \
    "$CUB_ROOT/attributes/attributes.txt" \
    "$(dirname "$CUB_ROOT")/attributes.txt" \
    "$CURATED_DATA/CUB_processed/attributes.txt"
  do
    if [ -s "$candidate" ]; then CUB_ATTR_FILE="$candidate"; break; fi
  done
fi
: "${CUB_ATTR_FILE:?could not locate attributes.txt; export CUB_ATTR_FILE}"

cd "$SOURCE_REPO"
bash curated/train/verify_canonical_sources.sh
if [ ! -s "$CANONICAL_ROOT/data/canonical_data_manifest.json" ]; then
  python3 curated/data/prepare_canonical_data.py \
    --repo "$SOURCE_REPO" --data-root "$CANONICAL_ROOT/data" \
    --funnybirds-root "$FUNNYBIRDS_ROOT" --cub-root "$CUB_ROOT" \
    --cub-attributes "$CUB_ATTR_FILE"
fi

# Freeze the submitted implementation. Pending jobs never observe later pulls,
# notebook execution, or edits in the user's normal checkout.
SNAPSHOT="$CANONICAL_ROOT/code"
SOURCE_SHA=$(git -C "$SOURCE_REPO" rev-parse HEAD)
if [ ! -d "$SNAPSHOT/.git" ] && [ ! -f "$SNAPSHOT/.git" ]; then
  git -C "$SOURCE_REPO" worktree add --detach "$SNAPSHOT" "$SOURCE_SHA"
  git -C "$SNAPSHOT" submodule update --init --recursive
fi
[ "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$SOURCE_SHA" ] || {
  echo "ERROR: canonical code snapshot exists at a different commit" >&2; exit 2;
}
REPO="$SNAPSHOT"
bash "$REPO/curated/train/verify_canonical_sources.sh"
export REPO CANONICAL_ROOT FUNNYBIRDS_ROOT CUB_ATTR_FILE
export CONFIRM_CANONICAL_REBUILD=YES
bash "$REPO/curated/train/submit_canonical_rebuild.sh"
