#!/usr/bin/env bash
# Read-only audit for the opt-in accelerated FunnyBird seed-1 protocol.
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"

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
grep -q -- '-log_dir "$OUT" -e 100 -optimizer sgd' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q -- '-attr_loss_weight 0.01 -normalize_loss -b 128' \
  "$CURATED/train/koh_joint_stage.sh"
grep -q -- '-weight_decay 0.0004 -lr 0.02' \
  "$CURATED/train/koh_joint_stage.sh"

echo "[KOH ACCELERATED PREFLIGHT PASS] original path preserved; accelerated path audited"
