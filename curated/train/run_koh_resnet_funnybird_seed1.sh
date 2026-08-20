#!/usr/bin/env bash
# Execute one approved FunnyBird Koh/ResNet seed-1 condition.  No submission.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
LABELS="${1:?usage: run_koh_resnet_funnybird_seed1.sh standard|rlv2}"
case "$LABELS" in standard|rlv2) ;; *) echo "ERROR: invalid labels $LABELS" >&2; exit 2 ;; esac
[ "${KOH_RESNET_RUN_APPROVED:-}" = YES ] || {
  echo "ERROR: set KOH_RESNET_RUN_APPROVED=YES only after reviewing the dry run" >&2
  exit 2
}

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
export REPO DATASET=funnybirds LABELS SEED=1 BACKBONE=resnet50
export KOH_OUTPUT_ROOT="${KOH_OUTPUT_ROOT:-$CURATED_DATA/koh_joint_resnet_v1}"
export KOH_RESTART_BACKUP_DIR="${KOH_RESTART_BACKUP_DIR:-$CURATED_DATA/koh_joint_resnet_restart_backup/funnybirds/$LABELS/seed1}"

exec bash "$REPO/curated/train/koh_joint_stage.sh"
