#!/usr/bin/env bash
# Rebuild and execute only the standard FunnyBird CBM report.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

cat >&2 <<'EOF'
INCOMPLETE: notebook 02 is disabled until the accepted FunnyBird Koh Joint
accelerated_v1 SUCCESS.json, final raw-z evaluation, and validated fixed swaps
exist and build_standard_cbm_reports.py has been migrated away from legacy
minimal_cbm CBM checkpoints. Nothing was executed.
EOF
exit 2
