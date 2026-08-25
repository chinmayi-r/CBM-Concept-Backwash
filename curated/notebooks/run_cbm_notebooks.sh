#!/usr/bin/env bash
# Execute and export the complete CBM notebook path in narrative order.
set -euo pipefail

: "${CURATED_DATA:?export CURATED_DATA}"
: "${CUB_ROOT:?export CUB_ROOT}"

cat >&2 <<'EOF'
INCOMPLETE: the combined CBM runner is disabled until notebooks 02 and 05 load
only validated official Koh Joint manifests. Its historical required-file list
accepts legacy minimal_cbm CBM outputs and must not be used for publication.
Nothing was executed.
EOF
exit 2
