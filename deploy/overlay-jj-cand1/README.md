# vllm-jj cand1 — tracked post-r38 Python overlay

The fix is pinned and version-controlled here (nothing lives only in a host tarball).

- Base image: `localinferencelab/vllm:jovian-judgement-community-20260914-r38` (newest published vendor tag)
- Base commit: `66c293578` ("r38"), tag `pin/r38-66c293578`
- Overlay source: vendor branch `dev/jovian-judgement` tip `8e1f1e587f`, mirrored here as `vendor/dev-jovian-judgement`, tag `pin/post-r38-8e1f1e587f`
- Overlaid: the 16 post-r38 files under `vllm/v1/{core/sched,worker,engine}/` (`overlay-files.txt`); full diff in `overlay.patch`
- Deliberately NOT overlaid: the other 124 changed files (MoE/attention/linear kernels under `vllm/model_executor/**`) — native-coupled, unsafe to mix with r38 native libs

## Deploy on another host

Requires the public vendor base image (43 GB, needed to serve the model anyway):

    docker pull localinferencelab/vllm:jovian-judgement-community-20260914-r38
    docker pull <OVERLAY_IMG>          # a few MB
    docker build -f Dockerfile.apply --build-arg OVERLAY=<OVERLAY_IMG> -t vllm-jj:cand1 .

Or fetch this directory from the repo and run `./build.sh`.

## Verify / roll back

    docker run --rm --entrypoint grep vllm-jj:cand1 -c num_invalid_spec_tokens \
      /opt/glm53-flash/vllm/vllm/v1/core/sched/output.py   # 2 = overlay applied, 1 = base

Rollback = run the vendor base tag again; nothing is mutated in place.
