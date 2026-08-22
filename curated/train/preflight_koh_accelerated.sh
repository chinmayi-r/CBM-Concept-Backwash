#!/usr/bin/env bash
# Read-only audit for the opt-in accelerated FunnyBird seed-1 protocol.
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
: "${CURATED_DATA:?export CURATED_DATA}"

python3 -m py_compile \
  "$CURATED/compat/koh_accelerated_training.py" \
  "$CURATED/compat/run_koh.py" \
  "$CURATED/analysis/audit_koh_accelerated.py" \
  "$CURATED/analysis/audit_koh_accelerated_convergence.py" \
  "$CURATED/analysis/test_koh_accelerated_restart.py" \
  "$CURATED/analysis/validate_koh_joint.py"
bash -n \
  "$CURATED/train/koh_joint_stage.sh" \
  "$CURATED/train/run_koh_accelerated_funnybird_seed1.sh" \
  "$CURATED/train/koh_accelerated_funnybird_seed1_job.slurm"

python3 "$CURATED/analysis/audit_koh_accelerated.py"
bash "$CURATED/train/preflight_koh_resnet.sh"

grep -q 'KOH_TRAINING_PROTOCOL:-koh_original' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q 'resnet50:funnybirds:standard:1' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q -- 'git apply --no-index --recount' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q -- '-log_dir "$OUT" -e 100 -optimizer sgd' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q -- '-attr_loss_weight 0.01 -normalize_loss -b 128' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q -- '-weight_decay 0.0004 -lr 0.02' \
  "$CURATED/train/koh_joint_stage.sh"

# Reproduce the production boundary: CURATED_DATA normally sits below the
# parent repository, so the isolated Koh copy is still beneath a Git worktree.
# Verify that --no-index patches the copy rather than discovering the parent.
mkdir -p "$CURATED_DATA/koh_joint_runtime"
PATCH_TEST="$(mktemp -d \
  "$CURATED_DATA/koh_joint_runtime/preflight_patch.XXXXXX")"
cleanup_patch_test() {
  case "$PATCH_TEST" in
    "$CURATED_DATA"/koh_joint_runtime/preflight_patch.*)
      rm -rf -- "$PATCH_TEST"
      ;;
    *)
      echo "ERROR: unsafe preflight cleanup target: $PATCH_TEST" >&2
      exit 2
      ;;
  esac
}
trap cleanup_patch_test EXIT
rsync -a --exclude=.git \
  "$CURATED/external/ConceptBottleneck/" "$PATCH_TEST/"
(cd "$PATCH_TEST" && git apply --no-index --recount \
  "$CURATED/patches/koh_restartable_training.patch")
grep -q 'koh_epoch_boundary_v1' "$PATCH_TEST/CUB/train.py"
if ! diff -qr --exclude=train.py \
  "$CURATED/external/ConceptBottleneck/CUB" "$PATCH_TEST/CUB" >/dev/null; then
  echo "ERROR: isolated patch test changed files other than CUB/train.py" >&2
  exit 2
fi
cleanup_patch_test
trap - EXIT

echo "[KOH ACCELERATED PREFLIGHT PASS] original path preserved; accelerated path audited"
