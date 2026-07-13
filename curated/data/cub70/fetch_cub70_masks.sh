#!/usr/bin/env bash
# Fetch the CUB70 part-segmentation dataset (Behzadi-Khormouji & Oramas, WACV 2023)
#   https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset  (MIT)
# Ships part_labels.txt (11 parts) + AnnotationMasksPerclass.tar.xz (masks by class,
# first 70 classes of the CUB-200-2011 TEST split). This script clones, extracts to
# $CURATED_DATA/cub70/masks, and PRINTS the resulting layout so we can point
# build_cub70_visibility.py at the real structure (the README doesn't state the
# per-file format). Run on adroit:  bash data/cub70/fetch_cub70_masks.sh
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA first}"
REPO_URL="https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset.git"
DEST="$CURATED_DATA/cub70"
SRC="$DEST/repo"
mkdir -p "$DEST"

echo "### [1/4] clone $REPO_URL"
if [ -d "$SRC/.git" ]; then
  echo "  already cloned -> $SRC (git pull)"; git -C "$SRC" pull --ff-only || true
else
  git clone --depth 1 "$REPO_URL" "$SRC"
fi

ARCHIVE="$SRC/AnnotationMasksPerclass.tar.xz"
[ -f "$ARCHIVE" ] || { echo "ERROR: $ARCHIVE not found after clone"; ls -la "$SRC"; exit 1; }

echo "### [2/4] check the archive is real (not a git-lfs pointer)"
if head -c 64 "$ARCHIVE" | grep -q "git-lfs"; then
  echo "  archive is an LFS pointer -> git lfs pull"
  ( cd "$SRC" && git lfs install && git lfs pull ) || {
    echo "ERROR: git-lfs needed but 'git lfs' unavailable. module load git-lfs (or conda install git-lfs) and re-run."; exit 1; }
fi
echo "  archive size: $(du -h "$ARCHIVE" | cut -f1)"

echo "### [3/4] extract -> $DEST/masks"
mkdir -p "$DEST/masks"
tar -xJf "$ARCHIVE" -C "$DEST/masks"
cp -f "$SRC/part_labels.txt" "$DEST/part_labels.txt" 2>/dev/null || true

echo "### [4/4] LAYOUT (paste this back so the visibility builder is pointed correctly)"
echo "--- top level under masks/ ---"
find "$DEST/masks" -maxdepth 1 | head -20
echo "--- first ~30 entries, depth<=3 ---"
find "$DEST/masks" -maxdepth 3 | head -30
echo "--- a sample leaf file + its type ---"
SAMPLE="$(find "$DEST/masks" -type f | head -1)"
echo "sample: $SAMPLE"
[ -n "$SAMPLE" ] && file "$SAMPLE" || true
echo "--- extensions present ---"
find "$DEST/masks" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head
echo
echo "DONE. Masks under $DEST/masks. Paste the LAYOUT block above; then run"
echo "  python data/cub70/build_cub70_visibility.py   (after it's adapted to this layout)"
