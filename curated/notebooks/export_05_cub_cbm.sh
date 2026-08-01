#!/bin/bash
# Export the already-executed notebook without rerunning any model analysis.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$HERE"

python analysis/finalize_05_notebook_metadata.py

# This nbconvert release's default Lab template drops alt metadata from PNG
# outputs.  The classic template preserves it, so the accessibility warning is
# genuinely fixed rather than hidden.
jupyter nbconvert --to html --template classic notebooks/05_cub_cbm.ipynb
