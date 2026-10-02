#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261002-7b398c9e420c91eb@sha256:2230db60afb4fd06dc2ef2b7f7f27dc7f78d7a50be70a08461b4df9fa5e90732}"
OVERLAY_IMG="${OVERLAY_IMG:-ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand6}"
FULL_IMG="${FULL_IMG:-lan-avelino/vllm-jj:jj-cand6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PIN="$(git -C "$HERE" rev-parse HEAD)"
if [ "$(uname -m)" != "x86_64" ]; then
  echo "refusing to build on $(uname -m): the overlay manifest must be amd64" >&2
  exit 1
fi
docker build -f "$HERE/Dockerfile.overlay-image" --label org.opencontainers.image.revision="$PIN" -t "$OVERLAY_IMG" "$HERE"
docker build -f "$HERE/Dockerfile.apply" --build-arg BASE="$BASE" --build-arg OVERLAY="$OVERLAY_IMG" --label org.opencontainers.image.revision="$PIN" -t "$FULL_IMG" "$HERE"
if [ "${PUSH:-0}" = "1" ]; then docker push "$OVERLAY_IMG"; if [ "${PUSH_FULL:-0}" = "1" ]; then docker push "$FULL_IMG"; fi; fi
echo "pin: $PIN"
