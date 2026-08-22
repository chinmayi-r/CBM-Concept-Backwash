#!/usr/bin/env bash
# Execute the declared accelerated Koh-architecture FunnyBird CBM. No submission.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
[ "${KOH_ACCELERATED_RUN_APPROVED:-}" = YES ] || {
  echo "ERROR: set KOH_ACCELERATED_RUN_APPROVED=YES after reviewing the dry run" >&2
  exit 2
}

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
export REPO
export DATASET=funnybirds
export LABELS=standard
export SEED=1
export BACKBONE=resnet50
export KOH_TRAINING_PROTOCOL=accelerated_v1
export KOH_OUTPUT_ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_resnet_accelerated_v1}"
export KOH_RESTART_BACKUP_DIR="${KOH_RESTART_BACKUP_DIR:-$CURATED_DATA/koh_joint_resnet_accelerated_restart_backup/funnybirds/standard/seed1}"

exec bash "$REPO/curated/train/koh_joint_stage.sh"
