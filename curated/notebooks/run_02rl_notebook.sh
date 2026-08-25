#!/usr/bin/env bash
# Build, execute, and export the CBM-only matched RLv2 causal notebook.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

cat >&2 <<'EOF'
INCOMPLETE: notebook 02rl is intentionally disabled until official Koh Joint
standard/RLv2 checkpoints and fixed-render evaluations replace the legacy
minimal_cbm CBM artifacts.
EOF
exit 2
