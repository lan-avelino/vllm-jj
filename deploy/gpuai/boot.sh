#!/usr/bin/env bash
# GPU.ai container entrypoint: DeepSeek-V4.1-Flash-1017-NVFP4 on the vllm-jj cand7 stack.
#
# The instance runs the Karmic Kraken base image (pinned digest) as its container. On every
# start this script applies the cand7 overlay, fetches the model (skipped once complete) and
# serves on $HOST:$PORT under a restart loop. GPU.ai's app URL fronts $PORT with basic auth,
# which occupies the Authorization header, so vLLM runs without --api-key and binds to
# loopback unless HOST says otherwise.
set -euo pipefail

BOOT_BASE=${BOOT_BASE:?raw URL of deploy/gpuai at a pinned commit}
DATA=${DATA:-/workspace}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SITE=/opt/venv/lib/python3.12/site-packages
MODEL_REPO=nvidia/DeepSeek-V4.1-Flash-1017-NVFP4
MODEL_REV=480550d349ede47a048eb65a276c5c01a06256e2
MODEL_DIR=$DATA/models/DeepSeek-V4.1-Flash-1017-NVFP4
OVERLAY_REPO=lan-avelino/vllm-jj-overlay
OVERLAY_LAYER=sha256:551c6b8a308997dbb19b1922496948b25b4b7b96077ea17083e809239f5c06a6
OVERLAY_COMMIT=78bb9ebee3b18da6da3f0ecb4bf5903e2579fd11

mkdir -p "$DATA/logs" "$DATA/cache" "$DATA/models" "$DATA/boot"
exec > >(tee -a "$DATA/logs/boot.log") 2>&1
log() { echo "$(date -u +%FT%TZ) boot: $*"; }

fetch() {
  curl -fsSL --retry 5 "$BOOT_BASE/$1" -o "$DATA/boot/$1"
  echo "$2  $DATA/boot/$1" | sha256sum -c --quiet -
}
fetch fetch_model.py e3a5af05a750994944c706ba1ffa1217d31e344618733ce65b6327b49fdeb57e
fetch serve.env c968a08a178d98036ee8542c4de0f3e8927278f5355b7cf2a53ab57ea57abcb2
fetch serve.argv.json 070706cd57aa4ca43a33b56e8f71d323b23849f81ff9f9d4c17dc8ea2b514164

if [ "$(cat "$SITE/.vllm-jj-overlay" 2>/dev/null)" != "$OVERLAY_COMMIT" ]; then
  log "verifying KK base files (vLLM 93dabce32f)"
  (
    cd "$SITE"
    printf '%s  %s\n' \
      81f987ff50353ed64076cb462036758765f6c007c3849c773b1b181f58c21414 vllm/v1/core/sched/scheduler.py \
      9807a36c740b874d012a8091890711599e1a4f5907ef9380ffc03adeb000f8fa vllm/v1/worker/gpu/structured_outputs.py \
      ccaaae9863bc004540510495251eb3e4ef0bb9aad6e021f620f504309d6ab4a6 vllm/envs.py \
      a6041ddf35d7aebb8452697ed16274451af533042acfc46760860725d153f5cb vllm/v1/structured_output/backend_xgrammar.py \
      913519ce6b80d0551a6ab6bd1bc5988777310b9d0c0c9745ce805dd004470e20 vllm/config/engram.py \
      be4efe504e362d3cc5aa0241234b547f865c148aff3b2a99e263b8c1c6e58fdd vllm/models/deepseek_v41/common/engram.py \
      9c644c1fed14c73b25455baa75d13b091a2431ba057865b180f0d3e03062efb0 vllm/models/deepseek_v41/nvidia/engram.py \
      bfc500c4989607809577cbd10512b96e9162a7359ad407f772b7f695eaa34cd9 vllm/models/deepseek_v41/quant_config.py \
      efda07bdd54d947852cdf75095905033038cf2a4425d7b234d092a9253c7c811 vllm/models/deepseek_v4_1/common/engram.py \
      0cb3bd8793ae4e377b634d4684c7c992af26d4c28261da4a13dc45c56e90e882 vllm/models/deepseek_v4_1/quant_config.py |
      sha256sum -c --quiet -
    test ! -e vllm/v1/worker/gpu/structured_outputs_warmup.py
  ) || { log "base files differ; refusing to overlay"; exit 1; }
  tok=$(curl -fsS "https://ghcr.io/token?scope=repository:$OVERLAY_REPO:pull" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
  tmp=$(mktemp -d)
  curl -fsSL --retry 5 -H "Authorization: Bearer $tok" -o "$tmp/layer.tgz" \
    "https://ghcr.io/v2/$OVERLAY_REPO/blobs/$OVERLAY_LAYER"
  echo "${OVERLAY_LAYER#sha256:}  $tmp/layer.tgz" | sha256sum -c --quiet -
  tar -xzf "$tmp/layer.tgz" -C "$tmp"
  cp -a "$tmp/overlay/." "$SITE/"
  rm -rf "$tmp"
  echo "$OVERLAY_COMMIT" > "$SITE/.vllm-jj-overlay"
  log "overlay $OVERLAY_COMMIT applied: whitespace cap markers=$(grep -c VLLM_XGRAMMAR_MAX_WHITESPACE_CNT "$SITE/vllm/envs.py"), packed engram markers=$(grep -c lookup_packed "$SITE/vllm/models/deepseek_v4_1/common/engram.py")"
fi

until python3 "$DATA/boot/fetch_model.py" "$MODEL_REPO" "$MODEL_REV" "$MODEL_DIR" >> "$DATA/logs/model.log" 2>&1; do
  log "model fetch failed, retrying in 60s (see $DATA/logs/model.log)"
  sleep 60
done
log "model ready at $MODEL_DIR"

ln -sfn "$MODEL_DIR" /model
ln -sfn "$DATA/cache" /cache

set -a
. "$DATA/boot/serve.env"
set +a
export HOME=/root DO_NOT_TRACK=1 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=150 VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1
mapfile -t ARGV < <(python3 - "$DATA/boot/serve.argv.json" "$HOST" "$PORT" <<'PY'
import json, sys
argv, host, port = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
out, i = [], 0
while i < len(argv):
    if argv[i] in ("--host", "--port"):
        out += [argv[i], host if argv[i] == "--host" else port]
        i += 2
    else:
        out.append(argv[i])
        i += 1
print("\n".join(out))
PY
)

ulimit -s 65536 || log "stack limit stays at $(ulimit -s)"
ulimit -l unlimited || log "memlock stays at $(ulimit -l)"

while true; do
  python3 -c '
import os, sys
for d, _, files in os.walk(sys.argv[1]):
    for f in files:
        fd = os.open(os.path.join(d, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)' "$MODEL_DIR"
  log "starting vLLM on $HOST:$PORT"
  set +e
  "${ARGV[@]}" >> "$DATA/logs/vllm.log" 2>&1
  rc=$?
  set -e
  log "vLLM exited with status $rc, restarting in 15s"
  sleep 15
done
