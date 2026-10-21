# GPU.ai deployment

GPU.ai exposes container ports only on custom-image launches, so the instance runs the
Karmic Kraken base image directly and `boot.sh` is its entrypoint. There is no proot here:
the image is the container.

On every container start `boot.sh`:

1. fetches its sibling files from `$BOOT_BASE` (this directory at a pinned commit) and
   checks each against the sha256 recorded in `boot.sh`;
2. verifies the four KK base files against the hashes in `../overlay-jj-cand6/Dockerfile.apply`,
   then applies the cand6 overlay layer (`vllm-jj-overlay` blob pinned by digest);
3. downloads `deepseek-ai/DeepSeek-V4.1-Flash-1017` at a pinned revision with
   `fetch_model.py` (resumable, sha256-verified; a no-op once complete);
4. serves on `$PORT` (default 8000) with the same env and argv as the gpuhub hosts, under a
   restart loop. Logs are in `$DATA/logs` (`boot.log`, `model.log`, `vllm.log`).

`serve.env` is the KK launcher's environment minus what the image already sets, and
`serve.argv.json` is the launcher argv (top_p 1.0, reasoning effort high), both taken from
the gpuhub deployment.

The app URL (`https://<id>.apps.gpu.ai`) is behind GPU.ai's per-instance basic auth, which
uses the `Authorization` header that vLLM's `--api-key` also needs, so vLLM runs without an
API key and the basic-auth login is the access control. vLLM binds to `127.0.0.1` so only the
in-container tunnel client can reach it; set `HOST` in the launch env only if GPU.ai's port
exposure cannot reach loopback.

## Launch

The CLI (1.3) has no image or port flags, so create the instance through the API:

```bash
BOOT_BASE=https://raw.githubusercontent.com/lan-avelino/vllm-jj/<commit>/deploy/gpuai
curl -sS https://api.gpu.ai/v1/instances -H "Authorization: Bearer $GPUAI_API_KEY" \
  -H 'Content-Type: application/json' -d @- <<EOF
{
  "gpu_type": "rtx_pro_6000", "gpu_count": 4, "tier": "on_demand", "region": "us-east",
  "image": "ghcr.io/local-inference-lab/vllm@sha256:2230db60afb4fd06dc2ef2b7f7f27dc7f78d7a50be70a08461b4df9fa5e90732",
  "entrypoint": ["bash", "-c", "set -e; mkdir -p /workspace/boot; curl -fsSL --retry 5 \"\$BOOT_BASE/boot.sh\" -o /workspace/boot/boot.sh; echo \"\$BOOT_SHA256  /workspace/boot/boot.sh\" | sha256sum -c -; exec bash /workspace/boot/boot.sh"],
  "env": {"BOOT_BASE": "$BOOT_BASE", "BOOT_SHA256": "<sha256 of boot.sh>"},
  "ports": [{"port": 8000, "protocol": "http", "primary": true}],
  "disk_gb": 1000,
  "ssh_key_ids": ["<key id>"]
}
EOF
```

`disk_gb` matters: the default 100 GB evicts the pod once the 510 GB model download passes it.
