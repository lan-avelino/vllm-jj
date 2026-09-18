# vllm-jj cand2 — vendor post-r38 fixes + custom empty-step liveness guard

**Deployment-custom. Not intended for upstream contribution.**

= `deploy/overlay-jj-cand1` (16 vendor post-r38 files from `dev/jovian-judgement` `8e1f1e587f`)
  **+ a local liveness guard** in `vllm/v1/core/sched/scheduler.py`

The guard enforces one invariant: the scheduler must never return an empty step
while runnable work exists. `schedule_running_requests()` gains `force=True`,
bypassing prefill-interleave lane selection, deferred prefills and the interleave
token-budget fan-out; the guard invokes it once per empty step and logs the gate
state (throttled) if even that produces nothing, so a genuine KV-exhaustion stall
is visible rather than silent.

- Base image: `localinferencelab/vllm:jovian-judgement-community-20260914-r38`
- Base commit: `66c293578`; vendor overlay source: `8e1f1e587f`
- Patch commit: pinned in `build.sh` (`PIN`), branch `deploy/jj-cand2`
- Regression test: `tests/v1/core/test_prefill_compute_share_scheduler.py::test_empty_step_liveness_guard_forces_progress`
  (passes with the guard, fails without it)

## Deploy on another host

    docker pull localinferencelab/vllm:jovian-judgement-community-20260914-r38
    docker pull ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand2      # a few MB
    docker build -f Dockerfile.apply --build-arg OVERLAY=ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand2 -t vllm-jj:cand2 .

## Verify / roll back

    docker run --rm --entrypoint grep vllm-jj:cand2 -c 'Azeus deployment patch' \
      /opt/glm53-flash/vllm/vllm/v1/core/sched/scheduler.py    # 1 = guard present, 0 = absent

Rollback = run the vendor base tag again; nothing is mutated in place.
