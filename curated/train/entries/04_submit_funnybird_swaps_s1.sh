#!/usr/bin/env bash
# Entry 4: submit fixed swaps; optionally wait for the live RLv2 job id.
set -euo pipefail
: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
RL_JOB_ID="${1:-}"
JOB=koh_fb_seed1_swaps
ROOT="$CURATED_DATA/koh_joint_resnet_accelerated_v1/funnybirds"
OUT="$CURATED_DATA/swap_koh_joint_resnet_accelerated_v1_seed1"
STANDARD="$ROOT/standard/seed1/SUCCESS.json"
RL="$ROOT/rlv2/seed1/SUCCESS.json"
test "$(git -C "$REPO" branch --show-current)" = claude/cbm-mcbm-validation-curated-efkd4y || {
  echo "ERROR: wrong branch" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || { echo "ERROR: tracked changes" >&2; exit 2; }

echo "===== ENTRY 4: FUNNYBIRD FIXED SWAPS S1 ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
sacct -u "$USER" --starttime now-30days -X --name="$JOB" \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End
python3 "$REPO/curated/analysis/canonical_manifest.py" verify --manifest "$STANDARD"
test ! -s "$OUT/SUCCESS.json" || { echo "ERROR: already complete: $OUT/SUCCESS.json" >&2; exit 2; }
! squeue -h -u "$USER" -n "$JOB" | grep -q . || { echo "ERROR: already queued: $JOB" >&2; exit 2; }

dependency=()
if [ -s "$RL" ]; then
  python3 "$REPO/curated/analysis/canonical_manifest.py" verify --manifest "$RL"
  echo "rlv2_source=accepted manifest"
elif [ -n "$RL_JOB_ID" ]; then
  state=$(squeue -h -j "$RL_JOB_ID" -o %T | awk 'NF {print $1; exit}')
  name=$(squeue -h -j "$RL_JOB_ID" -o %j | awk 'NF {print $1; exit}')
  [ "$name" = koh_accel_fb_rlv2_s1 ] || { echo "ERROR: job $RL_JOB_ID is $name, not RLv2 s1" >&2; exit 2; }
  case "$state" in RUNNING|PENDING|CONFIGURING|COMPLETING) ;; *) echo "ERROR: RLv2 job state=$state" >&2; exit 2 ;; esac
  scontrol show job -dd "$RL_JOB_ID" | grep -E 'JobId=|JobName=|JobState=|Command=|WorkDir='
  dependency=(--dependency="afterok:$RL_JOB_ID")
  echo "rlv2_source=job $RL_JOB_ID dependency=afterok"
else
  echo "ERROR: RLv2 is not accepted; pass its live job id as the sole argument" >&2
  exit 2
fi

echo "operation=fixed renderer one-part swaps"
echo "models=accepted standard s1 and RLv2 s1 Koh Joint ResNet-50"
echo "steps=render/reuse cache -> infer raw logits -> validate -> compare -> manifest"
echo "output=$OUT"
jid=$(sbatch --parsable --job-name="$JOB" "${dependency[@]}" \
  --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,CAMPAIGN=seed1" \
  "$REPO/curated/train/koh_funnybird_seed1_swaps_job.slurm")
echo "[ENTRY 4 SUBMITTED] job=$jid output=$OUT"
scontrol show job -dd "$jid" | grep -E \
  'JobId=|JobName=|JobState=|Dependency=|Command=|WorkDir=|TimeLimit=|Environment='
