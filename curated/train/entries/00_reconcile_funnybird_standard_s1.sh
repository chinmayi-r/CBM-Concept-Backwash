#!/usr/bin/env bash
# Entry 0: accept the already-completed FunnyBird standard seed-1 artifact.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
JOB=3357208
MANIFEST="$CURATED_DATA/koh_joint_resnet_accelerated_v1/funnybirds/standard/seed1/SUCCESS.json"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 0: FUNNYBIRD STANDARD S1 RECONCILIATION ====="
echo "gpu_job=none"
echo "source_slurm_job=$JOB"
echo "model=Koh Joint CBM; backbone=ResNet-50; protocol=accelerated_v1"
echo "action=validate completed checkpoints/evaluations, preserve original convergence result, write decision and manifest"
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -j "$JOB" -X \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End

if [ -s "$MANIFEST" ]; then
  python3 "$REPO/curated/analysis/canonical_manifest.py" verify \
    --manifest "$MANIFEST"
  echo "[ENTRY 0 ALREADY COMPLETE] $MANIFEST"
  exit 0
fi

python3 "$REPO/curated/analysis/reconcile_koh_accelerated_seed1.py" \
  --repo "$REPO" --curated-data "$CURATED_DATA"
python3 "$REPO/curated/analysis/canonical_manifest.py" verify \
  --manifest "$MANIFEST"
echo "[ENTRY 0 COMPLETE] $MANIFEST"
