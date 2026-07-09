#!/usr/bin/env bash
# Shared config generation for curated runs on the OFFICIAL minimal_cbm trainer.
# Sourced by mcbm_gamma_sweep.sh and run_baselines.sh so vanilla / cbm / mcbm all
# go through ONE substitution path (same backbone, same data) -> the only thing
# that differs between models is the head + gamma. That is what makes the
# CBM-vs-MCBM comparison clean (no backbone/preproc confound).
#
# Backbone is a single knob (ARCH env, default resnet50 ~ inception_v3 in size),
# so the crossed robustness grid {resnet50,inception_v3} x {cbm,mcbm} is one flag.

_dataset_paths() {   # sets PKLS_DIR IMGS_DIR ATTR_DIR for dataset $1
  local ds="$1"
  case "$ds" in
    funnybirds)
      PKLS_DIR="${FB_PKLS:-$CURATED_DATA/funnybirds_processed}"
      IMGS_DIR=""; ATTR_DIR="" ;;
    cub|cub70)
      PKLS_DIR="${CUB_PKLS:-$CURATED_DATA/CUB_processed/class_attr_data_10}"
      IMGS_DIR="${CUB_IMGS:-$CURATED_DATA/CUB_200_2011/images}"
      ATTR_DIR="${CUB_ATTR:-$CURATED_DATA/CUB_200_2011}" ;;
    *) echo "unknown dataset: $ds" >&2; return 1 ;;
  esac
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
      -e "s|__PKLS_DIR__|${PKLS_DIR}|g" \
      -e "s|__IMGS_DIR__|${IMGS_DIR}|g" \
      -e "s|__ATTR_DIR__|${ATTR_DIR}|g" "$tmpl" > "$out"
  if grep -oE '__[A-Z_]+__' "$out" | sort -u | grep .; then
    echo "ERROR: unsubstituted token(s) above in $out" >&2; return 1
  fi
  return 0
}
