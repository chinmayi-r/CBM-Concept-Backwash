#!/usr/bin/env bash
# Retained only to prevent an old command from silently submitting superseded work.
set -euo pipefail
echo "ERROR: this former bulk launcher is obsolete and submits nothing." >&2
echo "Run entries 05 and 06 explicitly, then entry 07 with their two job ids:" >&2
echo "  bash curated/train/entries/05_submit_funnybird_standard_convergence_s1.sh" >&2
echo "  bash curated/train/entries/06_submit_funnybird_rlv2_convergence_s1.sh" >&2
echo "  bash curated/train/entries/07_submit_funnybird_converged_swaps_s1.sh STANDARD_JOB_ID RLV2_JOB_ID" >&2
exit 2
