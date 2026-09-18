#!/usr/bin/env bash
# cand2 = vendor post-r38 python fixes + the deployment-custom liveness guard.
#   ./build.sh                build overlay + full derived image locally
#   PUSH=1 ./build.sh         also push the tiny overlay image
#   PUSH=1 PUSH_FULL=1 ./build.sh   also push the full image (~43 GB upload)
set -euo pipefail
PIN="${PIN:-f985ddb3537ba808e0854a5cf61ae4866bad3714}"
BASE="${BASE:-localinferencelab/vllm:jovian-judgement-community-20260914-r38}"
OVERLAY_IMG="${OVERLAY_IMG:-lan-avelino/vllm-jj-overlay:jj-cand2}"
FULL_IMG="${FULL_IMG:-lan-avelino/vllm-jj:jj-cand2}"
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
