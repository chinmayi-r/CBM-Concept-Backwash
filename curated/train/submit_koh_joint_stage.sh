#!/usr/bin/env bash
# Submit one clearly named dataset/label stage. No seed depends on another seed.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
DATASET="${1:?usage: submit_koh_joint_stage.sh funnybirds|cub70|cub standard|rlv2 [seeds]}"
LABELS="${2:?usage: submit_koh_joint_stage.sh funnybirds|cub70|cub standard|rlv2 [seeds]}"
SEEDS="${3:-1}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"

case "$DATASET:$LABELS" in
  funnybirds:standard|funnybirds:rlv2|cub70:standard|cub:standard) ;;
  *) echo "ERROR: unsupported stage $DATASET:$LABELS" >&2; exit 2 ;;
esac

echo "===== FRESH USER QUEUE ====="
if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
  echo "[DRY RUN] queue lookup omitted"
else
  squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"
fi
echo "===== SOURCE ====="
echo "repo=$(git -C "$REPO" rev-parse HEAD)"
echo "koh=$(git -C "$REPO/curated/external/ConceptBottleneck" rev-parse HEAD)"
python3 -m py_compile "$REPO/curated/compat/run_koh.py" \
  "$REPO/curated/analysis/validate_koh_joint.py"
bash -n "$REPO/curated/train/koh_joint_stage.sh" \
  "$REPO/curated/train/koh_joint_job.slurm"

for seed in $SEEDS; do
  case "$seed" in 1|2|3) ;; *) echo "ERROR: invalid seed $seed" >&2; exit 2 ;; esac
  if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
    jid="DRY_koh_${DATASET}_${LABELS}_s${seed}"
  else
    jid=$(sbatch --parsable --job-name="koh_${DATASET}_${LABELS}_s${seed}" \
      --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,DATASET=$DATASET,LABELS=$LABELS,SEED=$seed" \
      "$REPO/curated/train/koh_joint_job.slurm")
  fi
  echo "$DATASET $LABELS seed=$seed job=$jid"
done
