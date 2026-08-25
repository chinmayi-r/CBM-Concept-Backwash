#!/usr/bin/env bash
# Queue the complete seed-1-only campaign. Slurm/QOS provides the two-GPU cap.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${FB_STANDARD_JOB_ID:?set the live/complete FunnyBird standard seed-1 job id}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
EXPECTED_BRANCH=claude/cbm-mcbm-validation-curated-efkd4y

echo "===== FRESH QUEUE ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
echo "===== RELEVANT ACCOUNTING ====="
sacct -j "$FB_STANDARD_JOB_ID" -X \
  --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit,Start,End

[ "$(git -C "$REPO" branch --show-current)" = "$EXPECTED_BRANCH" ] || {
  echo "ERROR: campaign checkout must be on $EXPECTED_BRANCH" >&2; exit 2;
}
git -C "$REPO" diff --quiet --ignore-submodules=dirty -- || {
  echo "ERROR: campaign checkout has tracked changes" >&2; exit 2;
}

state=$(squeue -h -j "$FB_STANDARD_JOB_ID" -o %T | awk 'NF {print $1; exit}')
if [ -z "$state" ]; then
  state=$(sacct -n -j "$FB_STANDARD_JOB_ID" -X --format=State | awk 'NF {print $1; exit}')
fi
case "$state" in
  RUNNING|PENDING|CONFIGURING|COMPLETING|COMPLETED) ;;
  *) echo "ERROR: FunnyBird standard job $FB_STANDARD_JOB_ID state=$state" >&2; exit 2 ;;
esac
if [ "$state" = COMPLETED ]; then
  test -s "$CURATED_DATA/koh_joint_resnet_accelerated_v1/funnybirds/standard/seed1/SUCCESS.json" || {
    echo "ERROR: standard job completed without an accepted SUCCESS.json" >&2; exit 2;
  }
fi

for split in train val test; do
  test -s "$CURATED_DATA/koh_joint_inputs/funnybirds/rlv2/$split.pkl"
  test -s "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original/$split.pkl"
  test -s "$CURATED_DATA/CUB_processed/class_attr_data_10/$split.pkl"
done
for dataset in cub70 cub; do
  for file in selection/train.pkl selection/test.pkl final/test.pkl; do
    test -s "$CURATED_DATA/mcbm_seeded_v1_inputs/$dataset/standard/$file" || {
      echo "ERROR: missing MCBM input $dataset/$file" >&2; exit 2;
    }
  done
done

existing=(
  "$CURATED_DATA/koh_joint_resnet_accelerated_v1/funnybirds/rlv2/seed1/SUCCESS.json"
  "$CURATED_DATA/koh_joint_resnet_v1/cub70/standard/seed1/SUCCESS.json"
  "$CURATED_DATA/koh_joint_resnet_v1/cub/standard/seed1/SUCCESS.json"
  "$CURATED_DATA/swap_koh_joint_resnet_accelerated_v1_seed1/SUCCESS.json"
  "$CURATED_DATA/mcbm_seeded_v1/cub70/standard/g0/seed1/SUCCESS.json"
  "$CURATED_DATA/mcbm_seeded_v1/cub70/standard/g3/seed1/SUCCESS.json"
  "$CURATED_DATA/mcbm_seeded_v1/cub70/standard/g5/seed1/SUCCESS.json"
)
for gamma in 0 0p1 0p3 1 3 5; do
  existing+=("$CURATED_DATA/mcbm_seeded_v1/cub/standard/g$gamma/seed1/SUCCESS.json")
done
for manifest in "${existing[@]}"; do
  test ! -s "$manifest" || {
    echo "ERROR: campaign target already complete; reconcile before resubmission: $manifest" >&2
    exit 2
  }
done

python3 -m py_compile \
  "$REPO/curated/compat/run_koh.py" \
  "$REPO/curated/compat/koh_resnet.py" \
  "$REPO/curated/compat/koh_accelerated_training.py" \
  "$REPO/curated/analysis/audit_koh_resnet.py"
bash -n \
  "$REPO/curated/train/koh_joint_stage.sh" \
  "$REPO/curated/train/koh_joint_job.slurm" \
  "$REPO/curated/train/koh_accelerated_funnybird_seed1_job.slurm" \
  "$REPO/curated/train/mcbm_seeded_job.slurm" \
  "$REPO/curated/train/submit_seed1_campaign.sh"
python3 "$REPO/curated/analysis/audit_koh_resnet.py" weights
python3 "$REPO/curated/analysis/audit_koh_resnet.py" model \
  --koh-root "$REPO/curated/external/ConceptBottleneck" \
  --num-classes 70 --num-attributes 112
python3 "$REPO/curated/analysis/audit_koh_resnet.py" model \
  --koh-root "$REPO/curated/external/ConceptBottleneck" \
  --num-classes 200 --num-attributes 112
bash "$REPO/curated/train/verify_canonical_sources.sh"
git -C "$REPO/curated/external/ConceptBottleneck" apply --recount --check \
  "$REPO/curated/patches/koh_restartable_training.patch"

[ "${SUBMIT_APPROVED:-}" = YES ] || {
  echo "[DRY RUN ONLY] Set SUBMIT_APPROVED=YES to queue this exact seed-1 campaign."
  exit 0
}

submit() {
  local name=$1 dependency=$2 script=$3 exports=$4
  shift 4
  if squeue -h -u "$USER" -n "$name" | grep -q .; then
    echo "ERROR: duplicate live job $name" >&2; exit 2
  fi
  local dep=()
  [ -z "$dependency" ] || dep=(--dependency="afterok:$dependency")
  sbatch --parsable --job-name="$name" "${dep[@]}" "$@" \
    --export="ALL,REPO=$REPO,CURATED_DATA=$CURATED_DATA,$exports" "$script"
}

root_accel="$CURATED_DATA/koh_joint_resnet_accelerated_v1"
root_resnet="$CURATED_DATA/koh_joint_resnet_v1"

rl=$(submit koh_accel_fb_rlv2_s1 "" \
  "$REPO/curated/train/koh_accelerated_funnybird_seed1_job.slurm" \
  "LABELS=rlv2,KOH_OUTPUT_ROOT=$root_accel")
cub70=$(submit koh_resnet_cub70_s1 "$FB_STANDARD_JOB_ID" \
  "$REPO/curated/train/koh_joint_job.slurm" \
  "DATASET=cub70,LABELS=standard,SEED=1,BACKBONE=resnet50,KOH_TRAINING_PROTOCOL=koh_original,KOH_OUTPUT_ROOT=$root_resnet" \
  --time=1-00:00:00)
cub=$(submit koh_resnet_cub_s1 "$rl" \
  "$REPO/curated/train/koh_joint_job.slurm" \
  "DATASET=cub,LABELS=standard,SEED=1,BACKBONE=resnet50,KOH_TRAINING_PROTOCOL=koh_original,KOH_OUTPUT_ROOT=$root_resnet" \
  --time=1-00:00:00)
swaps=$(submit koh_fb_seed1_swaps "$FB_STANDARD_JOB_ID:$rl" \
  "$REPO/curated/train/koh_funnybird_seed1_swaps_job.slurm" "CAMPAIGN=seed1")

# CUB70 gamma 3/5 previously failed with non-finite first-batch losses. A new
# seeded gamma-0 control must finish first; gamma 3/5 use that identical recipe.
m70g0=$(submit m_cub70_std_g0_s1 "$cub70" \
  "$REPO/curated/train/mcbm_seeded_job.slurm" \
  "DATASET=cub70,LABELS=standard,GAMMA=0,SEED=1")
m70g3=$(submit m_cub70_std_g3_s1 "$m70g0" \
  "$REPO/curated/train/mcbm_seeded_job.slurm" \
  "DATASET=cub70,LABELS=standard,GAMMA=3,SEED=1")
m70g5=$(submit m_cub70_std_g5_s1 "$m70g0" \
  "$REPO/curated/train/mcbm_seeded_job.slurm" \
  "DATASET=cub70,LABELS=standard,GAMMA=5,SEED=1")

full_mcbm=()
for gamma in 0 0.1 0.3 1 3 5; do
  tag="${gamma//./p}"
  full_mcbm+=("$(submit "m_cub_std_g${tag}_s1" "$cub" \
    "$REPO/curated/train/mcbm_seeded_job.slurm" \
    "DATASET=cub,LABELS=standard,GAMMA=$gamma,SEED=1")")
done

printf '%-28s %s\n' \
  funnybird_standard "$FB_STANDARD_JOB_ID" funnybird_rlv2 "$rl" \
  cub70_standard "$cub70" full_cub_standard "$cub" funnybird_swaps "$swaps" \
  cub70_mcbm_gamma0 "$m70g0" cub70_mcbm_gamma3 "$m70g3" \
  cub70_mcbm_gamma5 "$m70g5" full_cub_mcbm "${full_mcbm[*]}"
echo "===== SEED-1 CAMPAIGN QUEUED ====="
squeue -u "$USER" -o "%.18i %.40j %.2t %.12M %.12l %R"
