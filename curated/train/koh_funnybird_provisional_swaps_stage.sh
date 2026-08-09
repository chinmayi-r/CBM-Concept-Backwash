#!/usr/bin/env bash
# Evaluate the preserved walltime-truncated seed-1 Koh Joint checkpoints.
# This stage is deliberately isolated from canonical SUCCESS roots.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
CURATED="$REPO/curated"
KOH="$CURATED/external/ConceptBottleneck"
STD="${PROVISIONAL_STANDARD_CHECKPOINT:-$CURATED_DATA/koh_joint_v1/funnybirds/standard/seed1/best_model_1.pth}"
RL="${PROVISIONAL_RL_CHECKPOINT:-$CURATED_DATA/koh_joint_v1/provisional_timeout/funnybirds_rlv2/seed1/best_model_1.pth}"
OUT="${PROVISIONAL_SWAP_OUT:-$CURATED_DATA/swap_koh_joint_provisional_timeout}"
CACHE="${KOH_RENDER_CACHE:-$CURATED_DATA/swap_fixed_v2_attempt2/render_cache}"
STD_DATA="$CURATED_DATA/koh_joint_inputs/funnybirds/standard"
RL_DATA="$CURATED_DATA/koh_joint_inputs/funnybirds/rlv2"
WORK="$CURATED_DATA/koh_joint_inputs/work/funnybirds"
NAMES="$CURATED_DATA/koh_joint_inputs/funnybird_concept_names.json"

test -s "$STD" || { echo "ERROR: missing provisional standard checkpoint: $STD" >&2; exit 2; }
test -s "$RL" || { echo "ERROR: missing provisional RLv2 checkpoint: $RL" >&2; exit 2; }
test ! -e "$OUT/SUCCESS.json" || {
  echo "ERROR: provisional stage refuses a canonical SUCCESS manifest: $OUT/SUCCESS.json" >&2
  exit 2
}
mkdir -p "$OUT"

note="walltime-truncated official Koh Joint checkpoint; provisional seed-1 evidence only"
python3 "$CURATED/analysis/validate_koh_joint.py" \
  --checkpoint "$STD" --koh-root "$KOH" --dataset funnybirds \
  --labels standard --seed 1 --num-classes 50 --num-attributes 26 \
  --manifest "$OUT/standard_checkpoint_INCOMPLETE.json" \
  --status INCOMPLETE --note "$note"
python3 "$CURATED/analysis/validate_koh_joint.py" \
  --checkpoint "$RL" --koh-root "$KOH" --dataset funnybirds \
  --labels rlv2 --seed 1 --num-classes 50 --num-attributes 26 \
  --manifest "$OUT/rlv2_checkpoint_INCOMPLETE.json" \
  --status INCOMPLETE --note "$note"

python3 "$CURATED/analysis/export_koh_eval.py" --koh-root "$KOH" \
  --checkpoint "$STD" --kind joint --data-pkl "$STD_DATA/test.pkl" \
  --work-dir "$WORK" --n-attributes 26 --names "$NAMES" \
  --out "$OUT/standard_seed1_test.parquet"
python3 "$CURATED/analysis/export_koh_eval.py" --koh-root "$KOH" \
  --checkpoint "$RL" --kind joint --data-pkl "$RL_DATA/test.pkl" \
  --work-dir "$WORK" --n-attributes 26 --names "$NAMES" \
  --out "$OUT/rlv2_seed1_test.parquet"

echo "===== PROVISIONAL STANDARD SEED 1 FIXED SWAPS ====="
KOH_CHECKPOINT="$STD" KOH_KIND=joint KOH_NAME=funnybirds-cbm \
  CONFIG_PREFIX=funnybirds-cbm GAMMAS=0 SEEDS=1 \
  SWAP_OUT="$OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
  bash "$CURATED/train/renderer_swap.slurm"

echo "===== PROVISIONAL RLV2 SEED 1 FIXED SWAPS ====="
KOH_CHECKPOINT="$RL" KOH_KIND=joint KOH_NAME=funnybirds-cbm-rlv2matched \
  CONFIG_PREFIX=funnybirds-cbm-rlv2matched GAMMAS=0 SEEDS=1 \
  SWAP_OUT="$OUT" RENDER_CACHE="$CACHE" SKIP_COMPARE=1 \
  bash "$CURATED/train/renderer_swap.slurm"

python3 "$CURATED/analysis/validate_fixed_swaps.py" --out "$OUT"
python3 "$CURATED/analysis/compare_fixed_rl.py" --out "$OUT" --rl-tag rlv2matched

python3 - "$OUT" "$STD" "$RL" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, standard, rlv2 = map(Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

manifest = {
    "status": "INCOMPLETE",
    "accepted_for": "provisional seed-1 walltime-truncated Koh Joint swap comparison",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "standard_checkpoint": str(standard),
    "standard_sha256": digest(standard),
    "rlv2_checkpoint": str(rlv2),
    "rlv2_sha256": digest(rlv2),
    "outputs": [
        str(out / "funnybirds-cbm-s1.csv"),
        str(out / "funnybirds-cbm-rlv2matched-s1.csv"),
        str(out / "fixed_rl_comparison_rlv2matched.csv"),
        str(out / "standard_seed1_test.parquet"),
        str(out / "rlv2_seed1_test.parquet"),
    ],
}
(out / "PROVISIONAL_INCOMPLETE.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(f"[PROVISIONAL SWAP PASS] {out / 'PROVISIONAL_INCOMPLETE.json'}")
PY
