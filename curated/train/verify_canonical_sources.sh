#!/usr/bin/env bash
# Require the exact two upstream implementations and the one declared patch.
set -euo pipefail
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
KOH="$REPO/curated/external/ConceptBottleneck"
MCBM="$REPO/curated/external/minimal_cbm"
PATCH="$REPO/curated/patches/minimal_cbm.patch"
UPGRADE_PATCH="$REPO/curated/patches/minimal_cbm_fp32_upgrade.patch"

EXPECTED_KOH=d6353f270702b92feb5b084a6fd065f891d583f8
EXPECTED_MCBM=9ba535c8d8e4a5b54e801a31d9db3d819d0910ab
[ "$(git -C "$KOH" rev-parse HEAD)" = "$EXPECTED_KOH" ] || {
  echo "ERROR: Koh source SHA differs from canonical SHA" >&2; exit 2;
}
[ "$(git -C "$MCBM" rev-parse HEAD)" = "$EXPECTED_MCBM" ] || {
  echo "ERROR: minimal_cbm source SHA differs from canonical SHA" >&2; exit 2;
}

expected=$(grep '^diff --git ' "$PATCH" | awk '{print $3}' | sed 's#^a/##' | sort)
if git -C "$MCBM" diff --quiet -- .; then
  git -C "$MCBM" apply "$PATCH"
else
  changed=$(git -C "$MCBM" diff --name-only | sort)
  [ "$changed" = "$expected" ] || {
    echo "ERROR: minimal_cbm has tracked changes outside the declared patch" >&2
    diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$changed") || true
    exit 2
  }
  if ! git -C "$MCBM" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    # One-time migration from the preceding exact patch. Do not restore or
    # overwrite anything: every old patched blob must match before applying
    # the train.py-only upgrade.
    while read -r file prefix; do
      actual=$(git -C "$MCBM" hash-object "$file")
      case "$actual" in "$prefix"*) ;; *)
        echo "ERROR: existing minimal_cbm changes are not the recorded prior patch: $file" >&2
        exit 2
      esac
    done <<'EOF'
src/datasets/__init__.py a63adca
src/experiments/base.py 01b51fa
src/datasets/cub200.py 2f62040
src/models/mcbm.py 6be561f
src/experiments/train.py ae3bcb9
EOF
    git -C "$MCBM" apply "$UPGRADE_PATCH"
    echo "[CANONICAL PATCH UPGRADED] prior exact patch -> FP32 opt-in patch"
  fi
fi
git -C "$MCBM" apply --reverse --check "$PATCH" >/dev/null || {
  echo "ERROR: the exact canonical minimal_cbm patch is not present" >&2; exit 2;
}
changed=$(git -C "$MCBM" diff --name-only | sort)
[ "$changed" = "$expected" ] || {
  echo "ERROR: minimal_cbm has tracked changes outside the declared patch" >&2
  diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$changed") || true
  exit 2
}
grep -q 'FUNNYBIRDS' "$MCBM/src/datasets/__init__.py"
grep -q 'dim_y.*max' "$MCBM/src/datasets/cub200.py"
grep -q 'delta_j = ' "$MCBM/src/models/mcbm.py"
grep -q 'non-finite training loss' "$MCBM/src/experiments/train.py"
grep -q 'MCBM_TRAINING_PRECISION' "$MCBM/src/experiments/train.py"
while read -r file prefix; do
  actual=$(git -C "$MCBM" hash-object "$file")
  case "$actual" in "$prefix"*) ;; *)
    echo "ERROR: patched source content mismatch: $file" >&2; exit 2 ;;
  esac
done <<'EOF'
src/datasets/__init__.py a63adca
src/experiments/base.py 01b51fa
src/datasets/cub200.py 2f62040
src/models/mcbm.py 6be561f
src/experiments/train.py f568197
EOF
echo "[CANONICAL SOURCE SUCCESS] Koh=$EXPECTED_KOH minimal_cbm=$EXPECTED_MCBM + exact declared patch"
