#!/usr/bin/env bash
# Submit the complete canonical training graph, breadth-first by seed.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${CUB_ROOT:?export CUB_ROOT}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CANONICAL_ROOT="${CANONICAL_ROOT:-$CURATED_DATA/canonical_20260806_v1}"
DATA_MANIFEST="$CANONICAL_ROOT/data/canonical_data_manifest.json"
if [ "${SUBMIT_DRY_RUN:-0}" != 1 ]; then test -s "$DATA_MANIFEST" || {
  echo "ERROR: canonical data manifest missing: $DATA_MANIFEST" >&2; exit 2;
}; fi
[ "${CONFIRM_CANONICAL_REBUILD:-}" = YES ] || {
  echo "ERROR: set CONFIRM_CANONICAL_REBUILD=YES after reviewing the matrix" >&2; exit 2;
}

cd "$REPO"
CANONICAL_REPO_SHA=$(git rev-parse HEAD)
echo "===== FRESH USER QUEUE ====="
if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
  echo "[DRY RUN] squeue omitted"
else
  squeue -u "$USER" -o "%.12i %.30j %.2t %.12M %R"
fi
echo "===== PINNED IMPLEMENTATIONS ====="
echo "repo=$(git rev-parse HEAD)"
echo "koh=$(git -C curated/external/ConceptBottleneck rev-parse HEAD)"
echo "minimal_cbm=$(git -C curated/external/minimal_cbm rev-parse HEAD)"
test -f curated/external/ConceptBottleneck/experiments.py
test ! -e curated/external/ConceptBottleneck/src/experiments.py
if [ "${SUBMIT_DRY_RUN:-0}" != 1 ]; then
  bash curated/train/verify_canonical_sources.sh
fi
python3 -m py_compile curated/compat/run_koh.py curated/analysis/canonical_manifest.py
bash -n curated/train/canonical_stage.sh curated/train/canonical_job.slurm

TRACK="$CANONICAL_ROOT/submitted_jobs.tsv"
mkdir -p "$CANONICAL_ROOT/manifests"
mkdir -p "$CANONICAL_ROOT/logs"
[ ! -s "$TRACK" ] || {
  echo "ERROR: submission ledger already exists; refusing duplicate submission: $TRACK" >&2
  exit 2
}
printf 'job_id\tseed\tframework\tdataset\tlabels\tstage\tgamma\tvariant\tdependency\n' > "$TRACK"

submit() {
  local seed="$1" framework="$2" dataset="$3" labels="$4" stage="$5"
  local gamma="$6" dependency="$7" time_limit="$8"
  local variant="${9:-}"
  local dcode
  case "$dataset" in funnybirds) dcode=fb ;; cub) dcode=c2 ;; cub70) dcode=c7 ;; *) dcode=xx ;; esac
  local name="c_${framework:0:1}${dcode}_${stage#*_}_${labels:0:1}_s${seed}"
  [ -z "$gamma" ] || name="c_m${dcode}_g${gamma//./p}_${labels:0:1}s${seed}"
  if [ -n "$gamma" ] && [ "$framework" = eval ]; then
    name="c_e${dcode}_g${gamma//./p}_${labels:0:1}s${seed}"
  fi
  [ -z "$variant" ] || name="c_e${dcode}_${variant}_${labels:0:1}s${seed}"
  local dep_args=()
  [ -z "$dependency" ] || dep_args=(--dependency="$dependency")
  local export_line="ALL,REPO=$REPO,CANONICAL_REPO_SHA=$CANONICAL_REPO_SHA,CANONICAL_ROOT=$CANONICAL_ROOT,CURATED_DATA=$CURATED_DATA,CUB_ROOT=$CUB_ROOT,STAGE=$stage,DATASET=$dataset,LABELS=$labels,SEED=$seed,MAX_REQUEUES=2"
  [ -z "$gamma" ] || export_line="$export_line,GAMMA=$gamma"
  [ -z "$variant" ] || export_line="$export_line,VARIANT=$variant"
  local jid
  if [ "${SUBMIT_DRY_RUN:-0}" = 1 ]; then
    jid="DRY_${name}_${gamma:-na}"
    echo "[DRY SUBMIT] $jid dependency=${dependency:-none}" >&2
  else
    jid=$(sbatch --parsable --job-name="$name" --time="$time_limit" \
      --output="$CANONICAL_ROOT/logs/%x_%j.out" --export="$export_line" \
      "${dep_args[@]}" curated/train/canonical_job.slurm)
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$jid" "$seed" "$framework" "$dataset" "$labels" "$stage" "$gamma" "$variant" "$dependency" >> "$TRACK"
  echo "$jid"
}

submit_seed() {
  local seed="$1" wave_dependency="$2"
  local all=()
  local fb_standard_ids=() fb_rl_ids=()
  for dataset_labels in funnybirds:standard funnybirds:rlv2 cub:standard cub70:standard; do
    IFS=: read -r dataset labels <<< "$dataset_labels"
    local concept independent joint joint_sig extract sequential
    concept=$(submit "$seed" koh "$dataset" "$labels" koh_concept "" "$wave_dependency" 12:00:00); all+=("$concept")
    independent=$(submit "$seed" koh "$dataset" "$labels" koh_independent "" "$wave_dependency" 04:00:00); all+=("$independent")
    joint=$(submit "$seed" koh "$dataset" "$labels" koh_joint "" "$wave_dependency" 12:00:00); all+=("$joint")
    joint_sig=$(submit "$seed" koh "$dataset" "$labels" koh_joint_sigmoid "" "$wave_dependency" 12:00:00); all+=("$joint_sig")
    local extract_dep="afterok:$concept"
    extract=$(submit "$seed" koh "$dataset" "$labels" koh_extract "" "$extract_dep" 04:00:00); all+=("$extract")
    sequential=$(submit "$seed" koh "$dataset" "$labels" koh_sequential "" "afterok:$extract" 04:00:00); all+=("$sequential")
    if [ "$dataset:$labels" = funnybirds:standard ]; then
      fb_standard_ids+=("$concept" "$independent" "$joint" "$joint_sig" "$extract" "$sequential")
    elif [ "$dataset:$labels" = funnybirds:rlv2 ]; then
      fb_rl_ids+=("$concept" "$independent" "$joint" "$joint_sig" "$extract" "$sequential")
    fi
    local variant train_dep eval_id
    for variant in independent sequential joint joint_sigmoid; do
      case "$variant" in
        independent) train_dep="afterok:$concept:$independent" ;;
        sequential) train_dep="afterok:$concept:$sequential" ;;
        joint) train_dep="afterok:$joint" ;;
        joint_sigmoid) train_dep="afterok:$joint_sig" ;;
      esac
      eval_id=$(submit "$seed" eval "$dataset" "$labels" eval_koh "" "$train_dep" 02:00:00 "$variant")
      all+=("$eval_id")
    done
  done
  for dataset_labels in funnybirds:standard funnybirds:rlv2 cub:standard cub70:standard; do
    IFS=: read -r dataset labels <<< "$dataset_labels"
    local cbm_control cbm_eval
    cbm_control=$(submit "$seed" mcbm "$dataset" "$labels" minimal_cbm_cbm "" "$wave_dependency" 06:00:00)
    all+=("$cbm_control")
    cbm_eval=$(submit "$seed" eval "$dataset" "$labels" eval_minimal_cbm_cbm "" "afterok:$cbm_control" 02:00:00)
    all+=("$cbm_eval")
    if [ "$dataset:$labels" = funnybirds:standard ]; then fb_standard_ids+=("$cbm_control"); fi
    if [ "$dataset:$labels" = funnybirds:rlv2 ]; then fb_rl_ids+=("$cbm_control"); fi
    for gamma in 0 0.1 0.3 1 3 5; do
      jid="$(submit "$seed" mcbm "$dataset" "$labels" mcbm "$gamma" "$wave_dependency" 06:00:00)"
      all+=("$jid")
      eval_id="$(submit "$seed" eval "$dataset" "$labels" eval_mcbm "$gamma" "afterok:$jid" 02:00:00)"
      all+=("$eval_id")
      if [ "$dataset:$labels" = funnybirds:standard ]; then fb_standard_ids+=("$jid"); fi
      if [ "$dataset:$labels" = funnybirds:rlv2 ]; then fb_rl_ids+=("$jid"); fi
    done
  done
  local std_dep rl_train_dep std_swap rl_swap
  std_dep=$(IFS=:; echo "afterok:${fb_standard_ids[*]}")
  std_swap=$(submit "$seed" eval funnybirds standard swap_all "" "$std_dep" 12:00:00)
  all+=("$std_swap")
  rl_train_dep=$(IFS=:; echo "${fb_rl_ids[*]}")
  rl_swap=$(submit "$seed" eval funnybirds rlv2 swap_all "" "afterok:${std_swap}:${rl_train_dep}" 12:00:00)
  all+=("$rl_swap")
  local joined
  joined=$(IFS=:; echo "${all[*]}")
  echo "$joined"
}

echo "===== SUBMIT SEED 1 BREADTH WAVE ====="
SEED1_IDS=$(submit_seed 1 "")
# afterany deliberately lets replication proceed even if one seed-1 branch has
# a deterministic error. Seeds 2 and 3 are independent peers, not a chain.
WAVE2_DEP="afterany:$SEED1_IDS"
echo "===== SUBMIT SEEDS 2 AND 3, INDEPENDENT AFTER SEED-1 WAVE ====="
SEED2_IDS=$(submit_seed 2 "$WAVE2_DEP")
SEED3_IDS=$(submit_seed 3 "$WAVE2_DEP")
FINAL_DEP="afterany:$SEED2_IDS:$SEED3_IDS"
submit 0 final funnybirds standard finalize "" "$FINAL_DEP" 01:00:00 >/dev/null

echo "[SUBMISSION COMPLETE] $TRACK"
column -t -s $'\t' "$TRACK" || cat "$TRACK"
echo "Monitor: bash curated/train/status_canonical_rebuild.sh"
