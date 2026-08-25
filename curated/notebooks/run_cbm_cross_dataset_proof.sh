#!/usr/bin/env bash
# Execute the primary standard-CBM comparison: FunnyBird notebook 02, then CUB70 notebook 05.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"

cat >&2 <<'EOF'
INCOMPLETE: notebooks 02 and 05 are intentionally disabled while their loaders
are migrated from legacy minimal_cbm CBM artifacts to official Koh Joint
manifests. Do not regenerate or publish these notebooks from legacy checkpoints.
EOF
exit 2
