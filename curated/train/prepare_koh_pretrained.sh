#!/usr/bin/env bash
# Cache the exact official Inception-v3 weights on the login node. Compute
# nodes have no external DNS, so every Koh job must find this file locally.
set -euo pipefail

cache_root="${TORCH_HOME:-$HOME/.cache/torch}"
target="$cache_root/hub/checkpoints/inception_v3_google-1a9a5a14.pth"
url="https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"

valid() {
  [ -s "$target" ] && [ "$(sha256sum "$target" | awk '{print substr($1,1,8)}')" = 1a9a5a14 ]
}

if valid; then
  echo "[KOH PRETRAINED READY] $target"
  exit 0
fi

mkdir -p "$(dirname "$target")"
tmp="$target.download.$$"
trap 'rm -f "$tmp"' EXIT
echo "Downloading official Koh/PyTorch Inception weights once on the login node..."
curl --fail --location --retry 5 --retry-delay 3 "$url" --output "$tmp"
prefix=$(sha256sum "$tmp" | awk '{print substr($1,1,8)}')
[ "$prefix" = 1a9a5a14 ] || {
  echo "ERROR: pretrained-weight hash prefix is $prefix, expected 1a9a5a14" >&2
  exit 2
}
mv "$tmp" "$target"
trap - EXIT
echo "[KOH PRETRAINED READY] $target"

