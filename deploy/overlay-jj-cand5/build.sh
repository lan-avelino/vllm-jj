#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-localinferencelab/vllm:jovian-judgement-community-20260914-r38}"
OVERLAY_IMG="${OVERLAY_IMG:-lan-avelino/vllm-jj-overlay:jj-cand5}"
FULL_IMG="${FULL_IMG:-lan-avelino/vllm-jj:jj-cand5}"
PIN="d96ee2124b895bca1d32ce9bde122653f3ca4e30"
HERE="$(cd "$(dirname "$0")" && pwd)"
docker build -f "$HERE/Dockerfile.overlay-image" -t "$OVERLAY_IMG" "$HERE"
docker build -f "$HERE/Dockerfile.apply" --build-arg BASE="$BASE" --build-arg OVERLAY="$OVERLAY_IMG" -t "$FULL_IMG" "$HERE"
if [ "${PUSH:-0}" = "1" ]; then docker push "$OVERLAY_IMG"; if [ "${PUSH_FULL:-0}" = "1" ]; then docker push "$FULL_IMG"; fi; fi
echo "pin: $PIN"
