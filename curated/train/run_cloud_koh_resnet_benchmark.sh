#!/usr/bin/env bash
# Five-epoch benchmark that preserves an exact restart state for continuation.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA to persistent storage}"
[ "${CLOUD_BENCHMARK_APPROVED:-}" = YES ] || {
  echo "ERROR: set CLOUD_BENCHMARK_APPROVED=YES only after reviewing this payload" >&2
  exit 2
}
LABELS="${1:-standard}"
case "$LABELS" in standard|rlv2) ;; *) echo "ERROR: invalid labels $LABELS" >&2; exit 2 ;; esac
REPO="${REPO:-$(git rev-parse --show-toplevel)}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi unavailable" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash "$REPO/curated/train/preflight_koh_resnet.sh"
python3 "$REPO/curated/analysis/audit_koh_resnet.py" weights

export KOH_RESNET_RUN_APPROVED=YES
export KOH_BENCHMARK_EPOCHS=5
set +e
bash "$REPO/curated/train/run_koh_resnet_funnybird_seed1.sh" "$LABELS"
status=$?
set -e
if [ "$status" -ne 75 ]; then
  echo "ERROR: benchmark expected controlled status 75, got $status" >&2
  exit "$status"
fi
echo "[READY TO RESUME] unset KOH_BENCHMARK_EPOCHS and run the same seed-1 wrapper"
