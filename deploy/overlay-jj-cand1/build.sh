#!/usr/bin/env bash
# Rebuild the tracked overlay from the pinned vendor commit, then build images.
#   ./build.sh                        build overlay + full derived image locally
#   PUSH=1 ./build.sh                 also push the tiny overlay image
#   PUSH=1 PUSH_FULL=1 ./build.sh     also push the full image (~43 GB upload)
set -euo pipefail
PIN="${PIN:-8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368}"
BASE="${BASE:-localinferencelab/vllm:jovian-judgement-community-20260914-r38}"
OVERLAY_IMG="${OVERLAY_IMG:-lan-avelino/vllm-jj-overlay:jj-cand1}"
FULL_IMG="${FULL_IMG:-lan-avelino/vllm-jj:jj-cand1-post-r38}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"

# Deterministic: file list is pinned, so extraction over the existing tree is enough.
mkdir -p "$HERE/overlay-root"
git -C "$ROOT" archive "$PIN" $(tr '\n' ' ' < "$HERE/overlay-files.txt") | tar -x -C "$HERE/overlay-root"

docker build -f "$HERE/Dockerfile.overlay-image" -t "$OVERLAY_IMG" "$HERE"
docker build -f "$HERE/Dockerfile.apply" --build-arg BASE="$BASE" --build-arg OVERLAY="$OVERLAY_IMG" -t "$FULL_IMG" "$HERE"

if [ "${PUSH:-0}" = "1" ]; then
  docker push "$OVERLAY_IMG"
  if [ "${PUSH_FULL:-0}" = "1" ]; then docker push "$FULL_IMG"; fi
fi
