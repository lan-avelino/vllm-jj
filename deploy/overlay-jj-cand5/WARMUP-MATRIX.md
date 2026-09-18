# cand5 warmup matrix — `_apply_grammar_bitmask_kernel`

Step-1 output (variant enumeration). Read-only investigation, 2026-09-18.
Host: Verda `az-ai-fin-01` (135.181.8.218), container `ds41-jovian-judgement`,
image `lan-avelino/vllm-jj:jj-cand4`, Triton 3.7.1, target `cuda` arch **120**.

## 1. Hook point (must run before monitor activation)

- `vllm/v1/worker/gpu_worker.py:865` → `warmup_kernels(...)` (`vllm/v1/worker/gpu/warmup.py:204`)
- `vllm/v1/worker/gpu_worker.py:904` → `# All warmup is done — start monitoring…` → `activate_jit_monitor(...)`

Anything compiled before that transition cannot trip the monitor. Do **not** add a
warmup call inside `apply_grammar_bitmask` or any request path — that is exactly
the 10:08:31 failure.

## 2. Specialization axes (from `trace_triton_kernel_specialization_args`)

```
trace(_apply_grammar_bitmask_kernel)
  -> ('logits_stride', 'bitmask_stride', 'vocab_size', 'MASK_STRIDE', 'BLOCK_SIZE')
constexprs: indices [9, 10]  (MASK_STRIDE, BLOCK_SIZE)
do_not_specialize: []        (nothing excluded)
```

Pointer args are not in this list; they are covered by pointer-alignment
specialization instead (see §3).

## 3. Live values and their specialization classes

| arg | source | live value | Triton class |
|---|---|---|---|
| `MASK_STRIDE` | `mask_stride=self.decode_query_len` (`model_runner.py:475`) | 8 (spec-decode, `num_speculative_tokens=7`) | constexpr, fixed |
| `BLOCK_SIZE` | hardcoded in `structured_outputs.py` | 8192 | constexpr, fixed |
| `logits_stride` | `logits.stride(0)` | `vocab_size` | divisible-by-16 |
| `vocab_size` | `self.vocab_size` | DeepSeek-V4.1 (not in `config.json`) | divisible-by-16 |
| `bitmask_stride` | `grammar_bitmask.shape[1]` = `cdiv(vocab, 32)` | ≈4040 | **generic** (not div-16) |
| all pointer args | torch buffers / slices | 256 B-aligned allocations | `tt.divisibility = 16` (aligned) |

All six cached/observed variants carry exactly one scalar class set
(`logits_stride`/`vocab_size` div-16, `bitmask_stride` generic), so the scalar
axis is **degenerate**: one representative each via
`triton_scalar_specialization_rep()`.

## 4. Cache census (the empirical half)

`TRITON_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllm37b3906a-…/triton`
(volume `ds41-cache`; 147 variant dirs total, 3 for this kernel)

| dir | signature | what it is |
|---|---|---|
| `7ST4JKS6B62E…` | has `%input_ids_ptr: !tt.ptr<i32>` + `%input_logits_indices_ptr: !tt.ptr<i64>`, stores=2, constant `-1` present | **invalid-draft variant** (cand4 layout) |
| `OY6HXRYCYMC3…` | no input pointers, stores=1 | **`None` path, cand4 layout** (`loc(...structured_outputs.py:136:0)` = current kernel) |
| `O2PO4NDDD66Q…` | identical signature to the above | **`None` path, cand3 layout** (`loc(...:126:0)`) — stale |

### Finding: the compile key is sensitive to the kernel's **line numbers**

`O2PO4N…` and `OY6HX…` are byte-identical apart from `loc(...)` debug locations
(line 126/154/161 vs 136/180/…). cand4 inserted ~10 lines *above* the kernel
definition, which shifted its line numbers and therefore produced a new cache
key even for the unchanged `None` path.

Consequences:
1. **Any overlay edit above the kernel invalidates the vendor/vendor-adjacent
   bake for that kernel** — not only edits to the kernel body.
2. A bake must be produced from **the exact file layout of the image that will
   run**; otherwise it is silently unused.

## 5. Predicted matrix

| # | variant | mechanism | note |
|---|---|---|---|
| 1 | `input_ids_ptr=None`, `input_logits_indices_ptr=None` | launch-based on the worker's real buffers (`logits_indices`, `grammar_bitmask`) + compile-only aligned descriptors | covers every ordinary tool/json request |
| 2 | both pointers **present** (invalid-draft path) | launch-based call with real tensors | reached only when a grammar terminates mid-draft (the cand4 fix path) |

**Predicted variant count N = 2.** Scalar classes and pointer alignment add no
further rows. Verification (step 3) must produce exactly N compile dirs from a
cold `TRITON_CACHE_DIR`, then show zero monitor events under
`--jit-monitor-mode error`.

## 6. Distribution consequence (urgent, independent of cand5)

Verda's variants already exist **in the volume**, so it compiles nothing today.
A host **without** this volume (RunPod, running the same `jj-cand4` image) has no
cand4-layout variant cached — so its first grammar/tool request compiles, and
under the vendor's baked `--jit-monitor-mode error`
(`/opt/glm53-flash/vllm/serve-ds41-flash.sh:451`) that is a **deterministic
engine death**, not a probabilistic one. RunPod's template must set
`--jit-monitor-mode=warn` (or receive a baked warmed cache) before it serves
tool/json traffic.
