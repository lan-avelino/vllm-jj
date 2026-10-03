# proot-noaccept

proot 5.4.1 with `accept`/`accept4` removed from its seccomp filter, for running
the vLLM images on hosts without Docker (AutoDL/gpuhub containers).

## Why

Stock proot traps `accept`/`accept4` with `FILTER_SYSEXIT` so it can rewrite peer
addresses of path-bound unix sockets. NCCL's non-blocking bootstrap retries
`accept()` in a tight loop while it waits for peers, so every retry costs two
round-trips through proot's single tracer thread. The first rank to reach a
communicator setup floods the tracer and the late ranks never get scheduled far
enough to arrive.

Symptoms seen on gpuhub 4×RTX PRO 6000 hosts (driver 595.71), all with GPUs idle,
the late ranks parked in `ptrace_stop` and proot at ~96% of one core:

- cand4: hang in `warmup_kernels` at the DSpark adaptive-verification broadcast;
- cand6: hang on the first request at the same broadcast (lazy TP communicator);
- cand6 with `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1`: hang in eager `split_group`
  on 2 of 3 boots (a SIGABRT of the spinning rank showed it in `split_group`,
  and the other ranks resumed the instant it died).

With this build all 6 boots came up (5 × cand6, 1 × cand4). Three of them ran
without the split-group flag, i.e. the lazy path that used to hang on every
first request, and those first requests answered in ~1.1–1.4 s.

## What changes

Only `accept()` on a path-bound unix socket is affected: the peer address it
returns is no longer translated to the guest path. Peer addresses of such
sockets are almost always unnamed, and a TCP, path-bound unix and abstract
socket round-trip behaves the same as stock.

## Build

```bash
./build.sh            # clones v5.4.1, checks commit d4b49e22, applies the patch, builds static
```

The binary that was tested and deployed (Ubuntu 22.04 on gpuhub, gcc 11,
libtalloc-dev 2.3.3) has sha256
`e8c899a6067a92e8e0b199d3c68db4b0aef5878e007b68b150ba009749cf07ae`. Builds are
not bit-reproducible: they embed the build directory and a build ID, so a rerun
of `build.sh` on the same host already gives a different hash. Trust the pinned
commit and the patch, not the binary hash, when rebuilding.
