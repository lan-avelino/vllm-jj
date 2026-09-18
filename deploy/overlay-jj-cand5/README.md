# cand5 — grammar-bitmask Triton JIT prewarm

**cand5 = cand4 + one new file + a small hook in a file cand4 already overlays.**
No vendor file is added to the overlay (see "Why not a vendor file" below).

## What it fixes

On 2026-09-18 10:08:31 UTC the Verda engine died: all four TP workers raised

```
RuntimeError: Triton kernel JIT compilation during inference: _apply_grammar_bitmask_kernel.
This causes a latency spike; consider extending warmup to cover this shape/config.
```

in `vllm/utils/jit_monitor.py` (mode `error`, baked into the vendor serve script
`/opt/glm53-flash/vllm/serve-ds41-flash.sh:451`) → EngineCore died → in-flight
requests 500 → container exit.

Cause: cand4 changed `_apply_grammar_bitmask_kernel`, and **Triton's cache key
covers the kernel's line numbers**, so any edit above the kernel invalidates the
baked `TRITON_CACHE_DIR` entry even for unchanged code paths. The kernel is not
part of the fork's warmup matrix, so the first request needing an uncached
variant compiled during inference.

Full derivation, the variant matrix (cand5 warms **2 variants per dtype**) and
the cache census are in [`WARMUP-MATRIX.md`](./WARMUP-MATRIX.md).

## Files overlaid (6)

| file | change |
|---|---|
| `vllm/v1/worker/gpu/structured_outputs_warmup.py` | **new** — builds and launches the variant matrix (never raises; `VLLM_DISABLE_GRAMMAR_BITMASK_WARMUP=1` skips) |
| `vllm/v1/worker/gpu/structured_outputs.py` | exports `GRAMMAR_BITMASK_BLOCK_SIZE`, keeps `self.vocab_size`, calls the warmup from `__init__` |
| `vllm/v1/core/sched/output.py`, `sched/scheduler.py`, `worker/gpu/model_runner.py`, `worker/gpu/sample/batch_shard.py` | unchanged from cand4 (vendor grammar fix + Azeus liveness guard) |

Hook ordering is what makes it safe: `StructuredOutputsWorker.__init__` runs
during runner construction, which is **before** `warmup_kernels()`
(`gpu_worker.py:~865`) and therefore before `activate_jit_monitor()`
(`gpu_worker.py:~906`).

## Why not a vendor file

The fork has a first-class warmup framework (`model_executor/warmup/jit_warmup*.py`,
per-model `*_triton_warmup.py`), but registering our kernel there would mean
overlaying a vendor file. cand2 proved that lesson: overlaying vendor files
without their full dependency closure fails to boot
(`AttributeError: 'MultiprocExecutor' object has no attribute 'b12x_warmup_control'`).
Both files we touch are already ours.

## Build

```bash
# on a GPU host (the full image needs the 12.9 GB vendor base)
./build.sh                       # builds overlay (FROM scratch) + full image
PUSH=1 ./build.sh                # also push the overlay image to GHCR
```

The overlay image is published as
`ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand5`
(digest `sha256:65443e2ab92287e52dd408519da419c2e408f7c48a0cb1bae5fefa6029952efb`,
855 bytes compressed) — public, anonymous pull verified for cand1-cand4.

> **Build it on an amd64 host** (or `docker buildx --platform linux/amd64`).
> Building this overlay on an arm64 Mac yields an arm64-only manifest, and an
> amd64 host then fails with `no match for platform in manifest: not found` even
> though the layer contents are correct — so verify the *platform*, not just the
> files. (Measured 2026-09-18: Mac build = arm64, rebuilt on the GPU host = amd64.)

## Verify (do this off-peak: every recreate = ~4.3 min downtime)

1. **Markers** — counts are exact (a partially applied overlay changes them).
   Paths are relative to `/opt/glm53-flash/vllm/` inside the container.

```bash
docker exec <container> bash -lc '
  P=/opt/glm53-flash/vllm
  grep -c num_invalid_spec_tokens $P/vllm/v1/core/sched/output.py               # 2
  grep -c num_invalid_spec_tokens $P/vllm/v1/core/sched/scheduler.py            # 7
  grep -c num_invalid_spec_tokens $P/vllm/v1/worker/gpu/model_runner.py         # 1
  grep -c num_invalid_spec_tokens $P/vllm/v1/worker/gpu/sample/batch_shard.py   # 2
  grep -c num_invalid_spec_tokens $P/vllm/v1/worker/gpu/structured_outputs.py   # 6
  grep -c "Azeus deployment patch" $P/vllm/v1/core/sched/scheduler.py           # 1 (liveness guard)
  grep -c "Azeus deployment patch" $P/vllm/v1/worker/gpu/structured_outputs.py  # 3
  grep -c "Negative keys mark" $P/vllm/v1/worker/gpu/structured_outputs.py      # 1 (cand4 kernel marker)
  grep -c "grammar-bitmask warmup" $P/vllm/v1/worker/gpu/structured_outputs_warmup.py   # 3
  grep -c GRAMMAR_BITMASK_BLOCK_SIZE $P/vllm/v1/worker/gpu/structured_outputs_warmup.py # 4
  grep -c GRAMMAR_BITMASK_BLOCK_SIZE $P/vllm/v1/worker/gpu/structured_outputs.py        # 2 (definition + use)
  grep -c b12x_warmup_control $P/vllm/v1/engine/core.py                         # 0 (native coupling absent)
'
```

2. **Cold-cache count check (the decisive one)** — expect exactly `4`:
   2 pointer variants (common path / invalid-draft path) × 2 warmup dtypes
   (bf16, fp16 — the dtype is part of the Triton signature). Measured
   2026-09-18: 4 variant dirs, **all four** of them grammar-kernel variants
   (nothing else is compiled), 1.25 s.

```bash
docker run --rm --gpus all --entrypoint bash <full-image> -lc '
  export TRITON_CACHE_DIR=/tmp/cold-warmup
  python3 -c "import torch; from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker as W; W(8, 129280, torch.device(\"cuda\"), 8, 1)"
  find /tmp/cold-warmup -name "*.ttgir" | wc -l                                  # 4
  find /tmp/cold-warmup -name "*_apply_grammar_bitmask_kernel.ttgir" | wc -l      # 4'
```

3. **Startup log** — one line per worker process:

```
grammar-bitmask warmup: compiled 4 variant(s) in X.XXs (dtypes=['bfloat16','float16'], vocab_size=..., mask_stride=8)
```

4. **Strict-mode run** — the vendor's own default (`--jit-monitor-mode error`),
   exercising tool calls *and* `json_schema` ending mid-draft (the invalid-draft
   path): require **zero** `during inference` events across the run. Then soak at
   `warn`, then repeat at `error`.

5. Unit test (not shipped in the image; run from a mounted source tree):
   `python -m pytest tests/v1/worker/test_grammar_bitmask_warmup.py -q` — pins the
   matrix and the constexpr drift guard. It cannot run on the Mac (no `tblib`/`vllm`),
   so run it in a container.

## Standing rule (this is the second time it bit us)

> Any overlay that touches a `@triton.jit` kernel — **or inserts lines above one** —
> invalidates the baked `TRITON_CACHE_DIR` entry for that kernel and must ship a
> warmup entry plus a cold-cache count check before it is deployed.

## Rollback

Recreate with the previous image (`DS41_IMAGE=lan-avelino/vllm-jj:jj-cand4` in
`/root/ds41.env`, then `./start.sh`). The warmup can also be disabled without a
rebuild via `VLLM_DISABLE_GRAMMAR_BITMASK_WARMUP=1` in the override's
`environment:` — note it is *not* read from `ds41.env` alone
(see MEMORY: env overrides must go in the compose `environment:` block).
