#!/usr/bin/env bash
# Shared config generation for curated runs on the OFFICIAL minimal_cbm trainer.
# Sourced by mcbm_gamma_sweep.sh and run_baselines.sh so vanilla / cbm / mcbm all
# go through ONE substitution path (same backbone, same data) -> the only thing
# that differs between models is the head + gamma. That is what makes the
# CBM-vs-MCBM comparison clean (no backbone/preproc confound).
#
# Backbone is a single knob (ARCH env, default resnet50). Primary result holds it
# FIXED across models. The optional robustness axis is CAPACITY (resnet18 vs
# resnet50, same family so only size varies) -- NOT family (resnet50~inception_v3
# in size, so that pairing tests nothing). See DECISIONS.md sec C.2.

# Prefer the val-split dir (<base>_trainval, built by data/make_val_split.py) so
# the trainer's per-epoch eval is on VAL, not test (standard model selection).
# Sets VALSPLIT=yes|no|override. Warns (once) if absent. Inline (no subshell) so
# VALSPLIT survives in the caller.
_pick_pkls() {   # $1=base ; $2=override-env-value -> sets PKLS_DIR VALSPLIT
  local base="$1" ovr="$2"
  if   [ -n "$ovr" ];               then PKLS_DIR="$ovr";            VALSPLIT=override
  elif [ -d "${base}_trainval" ];   then PKLS_DIR="${base}_trainval"; VALSPLIT=yes
  else                                   PKLS_DIR="$base";            VALSPLIT=no; fi
}

_dataset_paths() {   # sets PKLS_DIR IMGS_DIR ATTR_DIR VALSPLIT for dataset $1
  local ds="$1"
  case "$ds" in
    funnybirds)
      _pick_pkls "$CURATED_DATA/funnybirds_processed" "${FB_PKLS:-}"
      IMGS_DIR=""; ATTR_DIR="" ;;
    cub)
      _pick_pkls "$CURATED_DATA/CUB_processed/class_attr_data_10" "${CUB_PKLS:-}"
      IMGS_DIR="${CUB_IMGS:-${CUB_ROOT:-$CURATED_DATA/CUB_200_2011}/images}"
      ATTR_DIR="${CUB_ATTR:-$CURATED_DATA/CUB_processed}" ;;
    cub70)
      _pick_pkls "$CURATED_DATA/CUB_processed/class_attr_data_10_cub70_original" "${CUB_PKLS:-}"
      IMGS_DIR="${CUB_IMGS:-${CUB_ROOT:-$CURATED_DATA/CUB_200_2011}/images}"
      ATTR_DIR="${CUB_ATTR:-$CURATED_DATA/CUB_processed}" ;;
    *) echo "unknown dataset: $ds" >&2; return 1 ;;
  esac
  if [ "${VALSPLIT:-no}" = no ]; then
    echo "[warn] no val split for $ds -> per-epoch eval is on TEST (leaks if you epoch-select)." >&2
    echo "       build it: python data/make_val_split.py --pkls-dir <the ${ds} pkls dir>" >&2
  fi
  if [ "$ds" != funnybirds ]; then
    [ -d "$IMGS_DIR" ] || {
      echo "ERROR: image directory not found: $IMGS_DIR" >&2
      echo "       export CUB_ROOT=/path/to/CUB_200_2011" >&2
      return 1
    }
    [ -f "$ATTR_DIR/attributes.txt" ] || {
      echo "ERROR: attribute dictionary not found: $ATTR_DIR/attributes.txt" >&2
      echo "       run: bash data/cub70/prepare_all.sh" >&2
      return 1
    }
  fi
}

# gen_config TEMPLATE OUT DATASET ARCH [GAMMA]
# substitutes tokens, refuses fabricated schema, verifies nothing is left unfilled.
gen_config() {
  local tmpl="$1" out="$2" ds="$3" arch="$4" gamma="${5:-}"
  [ -f "$tmpl" ] || { echo "no template: $tmpl" >&2; return 1; }
  if grep -qE '(\bname:|manifest_dir:|\$\{oc\.env)' "$tmpl"; then
    echo "ERROR: $tmpl still uses the OLD fabricated schema (name:/manifest_dir:/oc.env)." >&2
    return 1
  fi
  _dataset_paths "$ds" || return 1
  sed -e "s|__ARCH__|${arch}|g" \
      -e "s|__GAMMA__|${gamma}|g" \
      -e "s|__BASE_LR__|${BASE_LR:-0.01}|g" \
      -e "s|__PKLS_DIR__|${PKLS_DIR}|g" \
      -e "s|__IMGS_DIR__|${IMGS_DIR}|g" \
      -e "s|__ATTR_DIR__|${ATTR_DIR}|g" "$tmpl" > "$out"
  if grep -oE '__[A-Z_]+__' "$out" | sort -u | grep .; then
    echo "ERROR: unsubstituted token(s) above in $out" >&2; return 1
  fi
  return 0
}
