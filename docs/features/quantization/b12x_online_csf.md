# Load native FP4 expert weights with CSF scale compression

Set `VLLM_B12X_MOE_FP4_CSF=1` to compress native MXFP4 or NVFP4 MoE block scales
losslessly during B12X weight preparation. This requires the paired B12X
`scale_compression="csf"` preparation API. The default is `0`; keep it at `0`
to load native scales without compression.

For Qwen3.8-Flash-Next NVFP4 on one 96 GiB RTX PRO 6000 Blackwell, with its
PLE/ngram table in host RAM:

```bash
VLLM_B12X_MOE_FP4_CSF=1 VLLM_PLE_CPU_OFFLOAD=1 \
  vllm serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision b797d2e1160b9596b2570e56c1d3590faa09d4ed \
  --quantization modelopt_mixed --load-format safetensors \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --moe-backend b12x --linear-backend b12x \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.96 \
  --max-model-len 262144 --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The PLE table needs about 26.82 GiB of host memory at TP1, in addition to
checkpoint loading and process memory. The encoder stages the expert scale
parameters in CPU memory and prepares one layer on the GPU at a time. FP4
weights remain on the GPU. Loading then releases the full source scale planes
and retains compressed scales plus a model-scoped shared reconstruction
buffer. The checkpoint files are read unchanged and no converted checkpoint
is written.

Use the standard `safetensors` loader for this configuration. Compression
runs at startup; it does not re-encode scales per request or training step.
Dense layers retain their existing quantization. Expert activation precision
also retains its native selection: MXFP8 for DS4.1 and BF16 for Kimi; the
compression flag does not force BF16 activations.

The implementation requires TP with PP1/DP1, no expert parallelism, and no
microbatch overlap. Separate model configurations own separate buffers.
Loading a model does not retain another model's scratch. MXFP4 supports the
existing CSF A8/A16 geometries; NVFP4 supports A4/A16. NVFP4 with A8 and
automatic precision switching are unsupported by this compression path.

For measurements and qualification scope, see the B12X
[`validation/csf/online-scales.md`](https://github.com/local-inference-lab/b12x/blob/feat/csf-online-scales/validation/csf/online-scales.md)
report. Usable concurrency also depends on KV and recurrent-state capacity;
a checkpoint fitting in VRAM does not by itself qualify every batch size or
context length.
