#!/usr/bin/env bash
# Cache torchvision's declared ResNet50 V1 weights before a network-isolated job.
set -euo pipefail
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
python3 "$REPO/curated/analysis/audit_koh_resnet.py" fetch-weights
