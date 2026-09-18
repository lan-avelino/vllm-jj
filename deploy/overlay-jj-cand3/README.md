# vllm-jj cand3 — stock r38 + deployment liveness guard (ONE file)

**Deployment-custom. Not for upstream.**

Supersedes `deploy/overlay-jj-cand2`, which failed to boot:

    AttributeError: 'MultiprocExecutor' object has no attribute 'b12x_warmup_control'
    RuntimeError: Engine core initialization failed

cand2 overlaid 16 vendor post-r38 files (`vllm/v1/{core/sched,worker,engine}/`), but post-r38
`vllm/v1/engine/core.py` calls `model_executor.b12x_warmup_control()`, which the vendor added in
`vllm/v1/executor/abstract.py` — outside that set. The vendor branch tip is not dependency-closed
against r38's native build (it also leans on excluded `vllm/config/*` and native-coupled
`vllm/distributed/device_communicators`), so the thin-overlay approach cannot carry it safely.

cand3 keeps the vendor's tested r38 artifact and changes exactly one file: the liveness guard.

- Base: `localinferencelab/vllm:jovian-judgement-community-20260914-r38` (commit `66c293578`)
- Patch: `deploy/jj-cand3`, guard commit `ea45990ec7`
- Guarded invariant: the scheduler never returns an empty step while runnable work exists.
  `schedule_running_requests(force=True)` bypasses lane selection, deferred prefills and the
  interleave token-budget fan-out; the guard fires once per empty step and logs the gate state
  (throttled) if even that fails.
- Regression test: `tests/v1/core/test_prefill_compute_share_scheduler.py::test_empty_step_liveness_guard_forces_progress`
  (passes patched, fails unpatched)

## Deploy on another host

    docker pull localinferencelab/vllm:jovian-judgement-community-20260914-r38
    docker pull ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand3
    docker build -f Dockerfile.apply --build-arg OVERLAY=ghcr.io/lan-avelino/vllm-jj-overlay:jj-cand3 -t vllm-jj:cand3 .

## Verify / roll back

    docker run --rm --entrypoint grep vllm-jj:cand3 -c 'Azeus deployment patch' \
      /opt/glm53-flash/vllm/vllm/v1/core/sched/scheduler.py    # 1 = guard present, 0 = absent
    docker run --rm --entrypoint grep vllm-jj:cand3 -c num_invalid_spec_tokens \
      /opt/glm53-flash/vllm/vllm/v1/core/sched/output.py       # 1 = stock r38 (vendor overlay absent)

Rollback = run the vendor base tag again.
