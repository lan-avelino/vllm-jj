# cand6 — cand5's safety patches on Karmic Kraken beta 20261002

**Deployment-custom. Not for upstream.**

cand6 moves the deployment from the Jovian Judgement r38 base to
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261002-7b398c9e420c91eb`
(digest `sha256:2230db60…0732`, vLLM `93dabce32f`) and carries the two
cand5 patches upstream does not have, plus a `json_schema` whitespace cap found
by the serving canary on KK.

## What changed from cand5

| cand5 piece | cand6 |
|---|---|
| Vendor grammar-invalid-draft fix (`num_invalid_spec_tokens`, 4 files) | **Dropped** — upstream in KK (same vendor commit, `29f6622592`) |
| Enriched `num_accepted <= num_draft_tokens` assert | **Dropped** — upstream in KK |
| Azeus empty-step liveness guard | **Re-ported** to KK's scheduler (`ed6a1b6f79`) — r38's `scheduler.py` is ~780 lines away, so it is not copied |
| Grammar-bitmask JIT prewarm | **Carried** unchanged (`8329125388`) — kernel signature and production dtypes identical to r38 |
| — | **New from upstream:** prefill-lane admission fix ([vllm #959](https://github.com/local-inference-lab/vllm/pull/959)), needed with `--max-parallel-prefills` ≥ 2 |
| — | **New (Azeus):** `json_schema` whitespace cap, `VLLM_XGRAMMAR_MAX_WHITESPACE_CNT` (default 32, `0` = off); see below |

The liveness guard's `force=True` pass skips only lane selection, prefill
deferral and the lane token budget. It still honours KK's new
`_ec_transfer_pending` check.

On KK the V2 runner builds `StructuredOutputsWorker` in `load_model()`, which
still precedes `compile_or_warm_up_model()` → `activate_jit_monitor()`
(`gpu_worker.py:~1110`), so the warmup compiles while compiles are allowed.

Upstream's release report qualifies DeepSeek V4.1 Flash TP4 / DSpark K7 on this
exact digest (startup, logprob probes, C1/C8/C16). Its throughput numbers are
from 600 W Workstation cards and are not comparable to our hosts.

### json_schema whitespace cap

KK ships xgrammar 0.2.6, whose JSON grammar allows `[ \n\r\t]*` between tokens
(r38's 0.2.5 allowed `[ \n\t]*`). When the model's plan conflicts with the schema,
for example its reasoning settles on `{"answer": true}` but the schema demands
`"yes"`/`"no"`, it can stall on `\r\n` after `{"answer":` until `max_tokens` and
return unparseable JSON. The canary saw this on 5 of 192 grammar requests (all
strict `json_schema`, never tool calls) while cand4 scored 160/160 on the same host.

The fix passes xgrammar's `max_whitespace_cnt` from `vllm/v1/structured_output/backend_xgrammar.py`,
so every whitespace run is bounded and the model is forced back onto the schema.
Measured on 3 × 96 strict `json_schema` requests with thinking on:

| build | valid | correct | runs that hit the cap | max completion tokens |
|---|---|---|---|---|
| cand4 (xgrammar 0.2.5) | 288/288 | 288/288 | n/a | 129 |
| cand6 + cap 32 | 288/288 | 288/288 | 3 | 124 |
| cand6 with xgrammar 0.2.5 swapped in | 288/288 | 288/288 | n/a | 150 |

Downgrading xgrammar also avoids the stall, but it puts KK's structural-tag tool
calling on a dependency set nobody tested and still allows unbounded `\n`, so the
cap was chosen. It also applies to 0.2.5, should cand5 want it.

## Files overlaid (5)

Into `/opt/venv/lib/python3.12/site-packages/` (KK installs vLLM as a package,
not under `/opt/glm53-flash/vllm`):

| file | change |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | liveness guard |
| `vllm/v1/worker/gpu/structured_outputs.py` | warmup hook, `GRAMMAR_BITMASK_BLOCK_SIZE`, `self.vocab_size` |
| `vllm/v1/worker/gpu/structured_outputs_warmup.py` | **new** — the variant matrix (`VLLM_DISABLE_GRAMMAR_BITMASK_WARMUP=1` skips) |
| `vllm/envs.py` | registers `VLLM_XGRAMMAR_MAX_WHITESPACE_CNT` |
| `vllm/v1/structured_output/backend_xgrammar.py` | passes `max_whitespace_cnt` to both `compile_json_schema` calls |

`Dockerfile.apply` checks the base's four original files against their sha256
(taken from the release's `runtime-files.json.gz`, equal to the `93dabce32f`
blobs) and fails the build if `BASE` points at anything else.

## Build

CI publishes the overlay image (amd64) on every push touching this directory
(`.github/workflows/jj-overlay-image.yml`), as
`ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand6` and `:jj-cand6-<commit sha>`.
Assemble the full image on the GPU host:

```bash
docker build -f Dockerfile.apply \
  --build-arg OVERLAY=ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand6-<sha> \
  -t lan-avelino/vllm-jj:jj-cand6 .
```

Or build both locally:

```bash
# on an amd64 host; build.sh refuses arm64 (cand5 lesson: arm64-only manifest)
./build.sh
PUSH=1 ./build.sh                # also push the overlay image to GHCR
```

## Deploy

KK uses its own launcher (`runtime.explicit` + a launch YAML, see the upstream
[DS4.1 guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md)),
not the r38 `serve-ds41-flash.sh` + `/root/ds41.env` flow, so the host's
compose/start script must be rewritten for cand6. Keep our deployment's
`--max-parallel-prefills` (the stock profile uses 1) and pass
`--jit-monitor-mode warn` for the canary. Switch to `error` only after a clean
canary (step 4 below).

## Verify

1. **Markers** — exact counts; `P=/opt/venv/lib/python3.12/site-packages`.

```bash
docker exec <container> bash -lc '
  P=/opt/venv/lib/python3.12/site-packages
  grep -c "Azeus deployment patch" $P/vllm/v1/core/sched/scheduler.py               # 1
  grep -c "force=True" $P/vllm/v1/core/sched/scheduler.py                           # 3 (stock: 0)
  grep -c admission_open $P/vllm/v1/core/sched/scheduler.py                         # 3 (#959, upstream)
  grep -c num_invalid_spec_tokens $P/vllm/v1/core/sched/scheduler.py                # 12 (upstream)
  grep -c num_invalid_spec_tokens $P/vllm/v1/worker/gpu/structured_outputs.py       # 6 (upstream)
  grep -c "Azeus deployment patch" $P/vllm/v1/worker/gpu/structured_outputs.py      # 3
  grep -c GRAMMAR_BITMASK_BLOCK_SIZE $P/vllm/v1/worker/gpu/structured_outputs.py    # 2
  grep -c GRAMMAR_BITMASK_BLOCK_SIZE $P/vllm/v1/worker/gpu/structured_outputs_warmup.py  # 4
  grep -c "grammar-bitmask warmup" $P/vllm/v1/worker/gpu/structured_outputs_warmup.py    # 3
  grep -c "Azeus deployment patch" $P/vllm/envs.py                                  # 1
  grep -c VLLM_XGRAMMAR_MAX_WHITESPACE_CNT $P/vllm/envs.py                          # 3
  grep -c max_whitespace_cnt $P/vllm/v1/structured_output/backend_xgrammar.py       # 2
'
```

2. **Cold-cache count check** — expect exactly `4` (2 pointer variants × bf16/fp16):

```bash
docker run --rm --gpus all --entrypoint bash <full-image> -lc '
  export TRITON_CACHE_DIR=/tmp/cold-warmup
  /opt/venv/bin/python -c "import torch; from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker as W; W(8, 129280, torch.device(\"cuda\"), 8, 1)"
  find /tmp/cold-warmup -name "*.ttgir" | wc -l                                  # 4
  find /tmp/cold-warmup -name "*_apply_grammar_bitmask_kernel.ttgir" | wc -l      # 4'
```

3. **Startup log** — one line per worker:
   `grammar-bitmask warmup: compiled 4 variant(s) in X.XXs (dtypes=['bfloat16', 'float16'], vocab_size=..., mask_stride=8)`

4. **Canary load** at `warn`, with our `--max-parallel-prefills`:
   - fill every slot with long prompts (≥ `max-num-seqs` + 8 concurrent ~150K-token
     requests) — must not wedge;
   - tool calls and `json_schema` ending mid-draft (the invalid-draft path);
   - require **zero** `during inference` JIT events and zero
     `empty scheduler step with runnable work` warnings; then repeat at `error`;
   - compare C1/C8 against cand5 on the same host.

   The scripts are in `canary/` (stdlib only, run them from any host that can
   reach the server):

   ```bash
   python3 canary/canary.py http://127.0.0.1:6106 <keyfile> all     # lanes, grammar, tps; must print CANARY PASS
   for i in 1 2 3; do python3 canary/jsontest2.py http://127.0.0.1:6106 <keyfile> cand6 96; done
   ```

   `jsontest2.py` must report `valid` and `correct` equal to `n` on every run.
   `ws_max` above 2 is fine as long as it never exceeds the cap.

5. **Tests** (not shipped in the image). Mount only `tests/` so `import vllm`
   resolves to the overlaid package; install `requirements/test` deps if missing:

```bash
docker run --rm --gpus all -v "$PWD/tests:/work/tests" -w /work --entrypoint bash <full-image> -lc '
  /opt/venv/bin/python -m pytest -q \
    tests/v1/core/test_prefill_compute_share_scheduler.py \
    tests/v1/worker/test_grammar_bitmask_warmup.py \
    tests/v1/structured_output/test_backend_xgrammar_max_whitespace.py'
```

   `test_empty_step_liveness_guard_forces_progress` (guard),
   `test_queued_prefills_leave_lanes_to_running_prefills` (#959) and
   `test_whitespace_run_is_capped_by_default` (cap; fails on stock KK) are the
   ones that matter here.

## Standing rule

> Any overlay that touches a `@triton.jit` kernel — **or inserts lines above one** —
> invalidates the baked `TRITON_CACHE_DIR` entry for that kernel and must ship a
> warmup entry plus a cold-cache count check before it is deployed.

## Rollback

Recreate with `lan-avelino/vllm-jj:jj-cand5` and the r38 start flow. The warmup
can be disabled without a rebuild via `VLLM_DISABLE_GRAMMAR_BITMASK_WARMUP=1` in
the compose `environment:` block, and the whitespace cap via
`VLLM_XGRAMMAR_MAX_WHITESPACE_CNT=0`.

## Hosts without Docker (proot)

On AutoDL/gpuhub containers the image runs flattened under proot. Use the
patched proot from `deploy/proot-noaccept/`: stock proot 5.4.1 can starve NCCL
communicator setup and hang startup or the first request.
