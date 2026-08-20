#!/usr/bin/env bash
# Read-only/source-isolated preflight.  Does not submit or train a real model.
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
SOURCE="$CURATED/external/ConceptBottleneck"
RUNTIME="$(mktemp -d)"
trap 'rm -rf -- "$RUNTIME"' EXIT

python3 -m py_compile \
  "$CURATED/compat/run_koh.py" \
  "$CURATED/compat/koh_resnet.py" \
  "$CURATED/analysis/audit_koh_resnet.py" \
  "$CURATED/analysis/test_koh_restart_state.py" \
  "$CURATED/analysis/validate_koh_joint.py" \
  "$CURATED/analysis/export_koh_eval.py"
bash -n \
  "$CURATED/train/koh_joint_stage.sh" \
  "$CURATED/train/koh_joint_job.slurm"
python3 "$CURATED/analysis/audit_koh_resnet.py" launcher

command -v rsync >/dev/null || { echo "ERROR: rsync is required" >&2; exit 2; }
rsync -a --exclude=.git "$SOURCE/" "$RUNTIME/"
(cd "$RUNTIME" && git apply --recount "$CURATED/patches/koh_restartable_training.patch")
grep -q "koh_epoch_boundary_v1" "$RUNTIME/CUB/train.py"
if ! diff -qr --exclude=train.py "$SOURCE/CUB" "$RUNTIME/CUB" >/dev/null; then
  echo "ERROR: preflight runtime differs outside CUB/train.py" >&2
  exit 2
fi

python3 "$CURATED/analysis/audit_koh_resnet.py" model --koh-root "$RUNTIME"
python3 "$CURATED/analysis/test_koh_restart_state.py" --koh-root "$RUNTIME"

# Verify the seed guard before any GPU command can exist.
set +e
guard_output=$(python3 "$CURATED/compat/run_koh.py" \
  --curated-num-classes 50 --curated-koh-root "$RUNTIME" \
  --curated-backbone resnet50 --curated-require-seed-one \
  CUB Joint --seed 2 2>&1)
guard_status=$?
set -e
if [ "$guard_status" -eq 0 ] || ! grep -q "seed-one guard rejected" <<<"$guard_output"; then
  echo "ERROR: seed-one guard did not reject seed 2" >&2
  echo "$guard_output" >&2
  exit 2
fi

echo "[KOH RESNET PREFLIGHT PASS] source boundary, structure, loss, restart, seed gate"
