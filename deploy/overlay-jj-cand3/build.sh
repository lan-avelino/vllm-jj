#!/usr/bin/env bash
# cand3 = stock r38 + the deployment liveness guard (single file).
set -euo pipefail
PIN="${PIN:-ea45990ec76e7d5fbdb746fed9ce32d9eb590e5f}"
BASE="${BASE:-localinferencelab/vllm:jovian-judgement-community-20260914-r38}"
OVERLAY_IMG="${OVERLAY_IMG:-lan-avelino/vllm-jj-overlay:jj-cand3}"
FULL_IMG="${FULL_IMG:-lan-avelino/vllm-jj:jj-cand3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
mkdir -p "$HERE/overlay-root"
git -C "$ROOT" archive "$PIN" $(tr '\n' ' ' < "$HERE/overlay-files.txt") | tar -x -C "$HERE/overlay-root"
docker build -f "$HERE/Dockerfile.overlay-image" -t "$OVERLAY_IMG" "$HERE"
docker build -f "$HERE/Dockerfile.apply" --build-arg BASE="$BASE" --build-arg OVERLAY="$OVERLAY_IMG" -t "$FULL_IMG" "$HERE"
if [ "${PUSH:-0}" = "1" ]; then
  docker push "$OVERLAY_IMG"
  if [ "${PUSH_FULL:-0}" = "1" ]; then docker push "$FULL_IMG"; fi
fi
