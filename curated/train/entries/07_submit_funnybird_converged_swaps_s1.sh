#!/usr/bin/env bash
# Entry 7: fixed swaps after both matched convergence continuations succeed.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
STANDARD_JOB_ID="${1:-}"
RLV2_JOB_ID="${2:-}"
JOB=koh_fb_converged_swaps_s1
ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_converged_v1/funnybirds"
OUT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_converged_v1_seed1"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 7: MATCHED CONVERGED FUNNYBIRD FIXED SWAPS S1 ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB" \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || { echo "ERROR: already queued: $JOB" >&2; exit 2; }

dependency_ids=()
for item in "standard:$STANDARD_JOB_ID" "rlv2:$RLV2_JOB_ID"; do
  labels=${item%%:*}; job_id=${item#*:}
  manifest="$ROOT/$labels/seed1/SUCCESS.json"
  if [ -s "$manifest" ]; then
    python3 "$REPO/curated/analysis/canonical_manifest.py" verify --manifest "$manifest"
    echo "$labels source=accepted convergence manifest"
  else
    [ -n "$job_id" ] || { echo "ERROR: pass live Standard and RLv2 continuation job ids" >&2; exit 2; }
    state=$(squeue -h -j "$job_id" -o %T | awk 'NF {print $1; exit}')
    name=$(squeue -h -j "$job_id" -o %j | awk 'NF {print $1; exit}')
    expected="koh_fb_${labels}_converge_s1"
    [ "$name" = "$expected" ] || { echo "ERROR: job $job_id is $name, expected $expected" >&2; exit 2; }
    case "$state" in RUNNING|PENDING|CONFIGURING|COMPLETING) ;; *) echo "ERROR: $labels job state=$state" >&2; exit 2 ;; esac
    dependency_ids+=("$job_id")
    echo "$labels source=job $job_id dependency=afterok"
  fi
done

dependency=()
if [ ${#dependency_ids[@]} -gt 0 ]; then
  joined=$(IFS=:; echo "${dependency_ids[*]}")
  dependency=(--dependency="afterok:$joined")
else
  joined=none
fi
echo "operation=fixed renderer one-part swaps"
echo "models=matched accepted Standard and RLv2 Koh Joint ResNet-50 continuations"
echo "dependency=$joined output=$OUT"
echo "steps=reuse validated render cache -> infer raw logits -> validate -> compare -> manifest"
echo "COMMAND: sbatch --job-name=$JOB dependency=$joined --export=ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,KOH_FUNNYBIRD_MODEL_ROOT=$ROOT,KOH_FUNNYBIRD_SWAP_OUT=$OUT curated/train/koh_funnybird_seed1_swaps_job.slurm"
jid=$(sbatch --parsable --job-name="$JOB" "${dependency[@]}" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,KOH_FUNNYBIRD_MODEL_ROOT=$ROOT,KOH_FUNNYBIRD_SWAP_OUT=$OUT" \
  "$REPO/curated/train/koh_funnybird_seed1_swaps_job.slurm")
echo "[ENTRY 7 SUBMITTED] job=$jid dependency=$joined output=$OUT"
