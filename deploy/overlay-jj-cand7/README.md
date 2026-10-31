# cand7 — cand6 + DeepSeek-V4.1-Flash-1017-NVFP4 support

**Deployment-custom. Not for upstream.**

cand7 is cand6 (same base: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261002-7b398c9e420c91eb`,
digest `sha256:2230db60…0732`, vLLM `93dabce32f`) plus what this branch needs to load
and serve [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)
(revision `480550d349ede47a048eb65a276c5c01a06256e2`) on SM120. Everything cand6
carries (liveness guard, grammar-bitmask prewarm, `json_schema` whitespace cap) is
unchanged; see `../overlay-jj-cand6/README.md` for those.

## What the checkpoint changes

Compared with `deepseek-ai/DeepSeek-V4.1-Flash-1017`, which cand6 already serves:

| tensors | base 1017 | NVFP4 export |
|---|---|---|
| `layers.0..39.ffn.experts.*` | MXFP4 (`I8` + `F8_E8M0` `.scale`) | **NVFP4**: `U8` + `F8_E4M3` group-16 `weight_scale`, `F32` `weight_scale_2` / `input_scale` |
| `mtp.*.ffn.experts.*` (DSpark draft) | MXFP4 | MXFP4 (ModelOpt `ignore: mtp.*`) |
| `layers.{1,14}.engram.embed.weight` | `F8_E4M3 [rows, 256]` | **MXFP4 packed** `U8 [rows, 128]` (`engram_embed_quant: mxfp4`) |
| attention, shared experts, head | block-32 FP8 | unchanged |

`w1`/`w3` `weight_scale_2` and `input_scale` are tied in the checkpoint (checked on
12 experts across layers 0, 7, 20, 39), so the fused w13 global scale is exact.

## Changes

| change | files | source |
|---|---|---|
| NVFP4 routed experts on the SM120 stack: `DeepseekV41FP8Config` (`deepseek_v41_fp8`, b12x) sends main-model experts of a `moe_quant_algo: NVFP4` export to `ModelOptNvFp4FusedMoE` (b12x NVFP4 backend, W4A4); experts matched by ModelOpt `ignore` stay `Mxfp4MoEMethod` | `vllm/models/deepseek_v4_1/quant_config.py`, `vllm/models/deepseek_v41/quant_config.py` (helpers moved into `DeepseekV4Nvfp4ExpertsMixin`) | **new (Azeus)**: upstream only has NVFP4 experts in the non-SM120 `deepseek_v41` package |
| Keep ignored MTP/DSpark experts on MXFP4 in the non-SM120 config | `vllm/models/deepseek_v41/quant_config.py` | vllm [#59655](https://github.com/vllm-project/vllm/pull/59655) (merged) |
| MXFP4-packed Engram tables | `vllm/config/engram.py`, `vllm/models/deepseek_v41/{common,nvidia}/engram.py`, `vllm/models/deepseek_v4_1/common/engram.py` | vllm [#61530](https://github.com/vllm-project/vllm/pull/61530) (open), ported to this branch's split layout and to the SM120 stack |

On SM120 the model class comes from `vllm.models.deepseek_v4_1`, whose Engram lookup
is b12x's FP8-only kernel. For packed tables `ParallelEngramEmbedding.lookup` uses the
#61530 Triton kernel (`_engram_lookup_kernel`, `PACKED_FP4=True`) instead and skips
the b12x `EngramLookup` preparation unit; FP8 tables still go through b12x. The
output contract is the same (`[T, 6144]` bf16, rows outside the TP shard zero).

Engram table modes with a packed table: `device` and `ram` are supported;
`table_memory: disk` (and `disk_resident_scales`) is rejected at config time. The
RAM budget charges 136-byte rows (128 + 8 scale bytes) instead of 264, so the two
tables need about 104 GB of host RAM in total instead of about 203 GB.

## Files overlaid (11)

cand6's five, plus:

| file | change |
|---|---|
| `vllm/config/engram.py` | `get_engram_embed_quant`, packed RAM budget, rejects disk + MXFP4 |
| `vllm/models/deepseek_v41/common/engram.py` | `PACKED_FP4` lookup kernel, packed table allocation and loader |
| `vllm/models/deepseek_v41/nvidia/engram.py` | packed pinned / DP-shared storage |
| `vllm/models/deepseek_v41/quant_config.py` | #59655, `DeepseekV4Nvfp4ExpertsMixin` |
| `vllm/models/deepseek_v4_1/common/engram.py` | packed tables on the SM120 stack, `lookup_packed` |
| `vllm/models/deepseek_v4_1/quant_config.py` | NVFP4 expert dispatch |

`Dockerfile.apply` checks all ten original base files against their `93dabce32f`
sha256 before copying.

## Build

Same as cand6, with `cand7` names. CI publishes
`ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand7` and `:jj-cand7-<commit sha>`.

```bash
docker build -f Dockerfile.apply \
  --build-arg OVERLAY=ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand7-<sha> \
  -t lan-avelino/vllm-jj:jj-cand7 .
```

## Verify

Nothing here has run on a GPU yet. Before serving traffic:

1. **Markers** (`P=/opt/venv/lib/python3.12/site-packages`, stock KK counts 0 for each),
   plus cand6's marker list:

```bash
grep -c PACKED_FP4 $P/vllm/models/deepseek_v41/common/engram.py                    # 3
grep -c packed_fp4 $P/vllm/models/deepseek_v41/nvidia/engram.py                    # 10
grep -c lookup_packed $P/vllm/models/deepseek_v4_1/common/engram.py                # 2
grep -c _is_nvfp4_expert_layer $P/vllm/models/deepseek_v4_1/quant_config.py        # 1
grep -c DeepseekV4Nvfp4ExpertsMixin $P/vllm/models/deepseek_v41/quant_config.py    # 2
grep -c get_engram_embed_quant $P/vllm/config/engram.py                            # 2
```

2. **Tests** (mount `tests/` as in cand6):

```bash
/opt/venv/bin/python -m pytest -q \
  tests/quantization/test_fp8.py -k "deepseek_v41" \
  tests/kernels/test_engram.py tests/models/test_deepseek_v4_1_engram.py
/opt/venv/bin/python -m pytest -q tests/distributed/test_engram_dp_shard.py   # 4 GPUs
/opt/venv/bin/python -m pytest -q tests/kernels/moe/test_b12x.py -k nvfp4
```

   The b12x NVFP4 MoE kernel has only been tested at intermediate 640. Per-rank
   intermediate here is 576 (TP 4), which is aligned for every padding and swizzle
   step in vLLM. Startup and warmup are the real check.

3. **Startup** log (per worker; `info_once`, so one line each):
   - `Using 'B12X' NvFp4 MoE backend` (main experts);
   - `Using 'B12X_MXFP4_MXFP8' Mxfp4 MoE backend.` (DSpark draft experts, as on cand6);
   - two `Engram layer … allocating … mapped-host RAM (packed E2M1 weights and E8M0 scales)`.

4. **Canary** at `--jit-monitor-mode warn`, as for cand6 (`canary/`). This overlay
   edits files that hold `@triton.jit` kernels (`_engram_lookup_kernel` and the
   kernels below the edits), so it invalidates their baked Triton cache entries.
   They compile in the first dummy forward, before the JIT monitor activates.
   Require zero `during inference` JIT events before switching to `error`.

5. **Quality**: NVFP4 experts run W4A4 with static `input_scale`, so outputs differ
   from the MXFP4 base. Run GSM8K (`tests/evals/gsm8k/gsm8k_eval.py`) against the
   base 1017 on the same host, and compare DSpark acceptance length.

6. **Memory**: NVFP4's group-16 FP8 scales are larger than MXFP4's group-32 E8M0
   scales, so expert weights take more VRAM. Check the KV cache size at
   `--gpu-memory-utilization 0.95` against the base.

## Rollback

Serve the base `deepseek-ai/DeepSeek-V4.1-Flash-1017` on cand6. cand7 runs that
checkpoint unchanged too (no `moe_quant_algo`, no `engram_embed_quant`).
