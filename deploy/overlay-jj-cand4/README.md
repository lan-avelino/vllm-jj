# vllm-jj cand4 — vendor grammar/spec-decode termination fix, 5-file overlay on stock r38

**Deployment-custom. Not for upstream.**

Root cause this addresses: the scheduler already computes
`SchedulerOutput.num_invalid_spec_tokens` in r38 and even reads it in its own accounting
assertion, but the value was never propagated to the v2 worker / sharded sampler. The grammar
mask was therefore applied to draft tokens that had already been invalidated, so accepted
draft counts could exceed grammar-valid counts after termination -> in-flight output
placeholders desynced -> the scheduler's async "already at max_tokens" gate latched requests
in RUNNING forever -> the empty-step wedge (Running pinned at max_seq, Waiting growing, GPUs
idle, /health still 200).

Ported from vendor commit `8e1f1e587f` ("Fix deferred grammar rejection in v2 speculative
sampling") as an exactly-scoped overlay. The vendor branch tip also carries
`9c27ec090d` + `ab03e87100` (b12x/native startup stages) which require native symbols
(`current_workspace_manager`, `reserve_all`, `b12x_warmup_control`) absent from r38 - those are
what broke the earlier cand2 boot, and they are NOT needed here (`model_runner.py` needed
exactly one line from this commit).

Overlaid files (r38 + this fix):
- vllm/v1/core/sched/output.py (+2: GrammarOutput.num_invalid_spec_tokens)
- vllm/v1/core/sched/scheduler.py (+14/-2 fix, plus the Azeus liveness guard)
- vllm/v1/worker/gpu/model_runner.py (+1: thread it into the sampler)
- vllm/v1/worker/gpu/sample/batch_shard.py (+7: shard per sampler)
- vllm/v1/worker/gpu/structured_outputs.py (+19: negative mapping keys + restore invalid
  input IDs inside the existing grammar-mask Triton kernel, no extra GPU launch)

## Evidence (r38 image, real GPUs)

| Run | Result |
|---|---|
| vendor test `test_gpu_sampler_rejects_drafts_after_grammar_termination` on **stock r38** (5th arg stripped) | 12 failed / 20 passed - fails exactly on `num_verified > num_valid` |
| same test with this overlay | **32 passed** |
| `test_shard_grammar_output` with overlay | 2 passed |
| pre-existing tests in that file | 18 passed |
| scheduler + fairness suites (guard included) | see repo notes; 13 known GPU-only CLI failures |

## Deploy

    docker pull localinferencelab/vllm:jovian-judgement-community-20260914-r38
    docker pull ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand4
    docker build -f Dockerfile.apply --build-arg OVERLAY=ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand4 -t vllm-jj:cand4 .

## Verify

    grep -c num_invalid_spec_tokens .../vllm/v1/worker/gpu/structured_outputs.py   # 6
    grep -c 'Azeus deployment patch' .../vllm/v1/core/sched/scheduler.py           # 1
    grep -c current_workspace_manager .../vllm/v1/worker/gpu/model_runner.py       # 0 (native coupling absent)
