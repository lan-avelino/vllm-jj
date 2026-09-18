# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Azeus deployment patch: pre-compile the grammar-bitmask kernel variants.

Why this exists
---------------
The fork activates its Triton JIT monitor only after warmup completes
(``vllm/v1/worker/gpu_worker.py``: ``warmup_kernels()`` at line ~865, then
``activate_jit_monitor()`` at line ~906), and this deployment runs that monitor
in ``error`` mode, where any Triton compilation during inference raises and kills
the engine (observed 2026-09-18 10:08:31 UTC: all four TP workers died on
``_apply_grammar_bitmask_kernel``).

``_apply_grammar_bitmask_kernel`` is not covered by the fork's warmup matrix, and
Triton's cache key covers the kernel's *line numbers*, so any overlay edit above
the kernel -- even one that only shifts it -- invalidates the baked entry in
``TRITON_CACHE_DIR``. The first grammar/tool request after such a deploy would
therefore compile during inference.

This module compiles the whole specialization matrix during worker construction,
which happens before ``activate_jit_monitor()``. The derivation and the observed
compile keys are documented in ``deploy/overlay-jj-cand5/WARMUP-MATRIX.md``:

* ``MASK_STRIDE`` and ``BLOCK_SIZE`` are constexprs (``worker.mask_stride`` and
  ``GRAMMAR_BITMASK_BLOCK_SIZE``);
* ``logits_stride`` (== ``vocab_size``) and ``vocab_size`` are divisible by 16,
  ``bitmask_stride`` (``cdiv(vocab_size, 32)``) is not;
* every pointer is 16-byte aligned, so only the aligned variant exists;
* the only remaining axis is whether the two input pointers are present
  (invalid-draft path) or ``None`` (the common path) -> two variants per dtype.

Set ``VLLM_DISABLE_GRAMMAR_BITMASK_WARMUP=1`` to skip the warmup (escape hatch).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

DISABLE_ENV = "VLLM_DISABLE_GRAMMAR_BITMASK_WARMUP"
# The engine runs bf16 logits; fp16 is cheap insurance in case a config selects
# it. Unused variants only cost one compile each at load time.
DEFAULT_WARMUP_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
# One row is enough: the compile key depends on dtypes, pointer alignment
# classes and constexprs -- not on the grid or the tensor shapes.
WARMUP_ROWS = 1

# (grid, positional kernel args, constexpr kwargs)
Launch = tuple[tuple[int, ...], tuple[Any, ...], dict[str, Any]]


def _is_disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "0") not in ("", "0", "false", "False")


def build_warmup_launches(worker: Any, dtype: torch.dtype) -> list[Launch]:
    """Argument sets covering every variant of ``_apply_grammar_bitmask_kernel``.

    ``worker`` only needs the attributes used by ``StructuredOutputsWorker``:
    ``device``, ``vocab_size``, ``mask_stride``, ``logits_indices`` and
    ``grammar_bitmask``.
    """
    from vllm.v1.worker.gpu.structured_outputs import GRAMMAR_BITMASK_BLOCK_SIZE

    rows = max(1, min(WARMUP_ROWS, int(worker.logits_indices.shape[0])))
    vocab_size = int(worker.vocab_size)
    # Same dtype/stride classes as the production tensor built by the sampler.
    logits = torch.zeros((rows, vocab_size), dtype=dtype, device=worker.device)
    # Production slices its own buffers, so reuse them (identical strides).
    logits_indices = worker.logits_indices[:rows]
    cu_num_logits = torch.arange(rows + 1, dtype=torch.int32, device=worker.device)
    bitmask = worker.grammar_bitmask[:rows]
    grid = (rows, cdiv(vocab_size, GRAMMAR_BITMASK_BLOCK_SIZE))
    kwargs = {
        "MASK_STRIDE": int(worker.mask_stride),
        "BLOCK_SIZE": GRAMMAR_BITMASK_BLOCK_SIZE,
    }
    common = (logits, logits.stride(0), logits_indices, cu_num_logits)
    tail = (bitmask, bitmask.stride(0), vocab_size)

    # Variant 1: no invalid drafts -> both input pointers are None.
    none_variant: Launch = (grid, common + (None, None) + tail, kwargs)

    # Variant 2: invalid-draft path -> both pointers are real tensors. Logits
    # indices stay non-negative so the kernel's `-1` store is inert, and the
    # index it would write is in bounds anyway.
    input_ids = torch.zeros(rows, dtype=torch.int32, device=worker.device)
    input_logits_indices = torch.zeros(rows, dtype=torch.int64, device=worker.device)
    tensor_variant: Launch = (
        grid,
        common + (input_ids, input_logits_indices) + tail,
        kwargs,
    )
    return [none_variant, tensor_variant]


def run_warmup(
    worker: Any, dtypes: Sequence[torch.dtype] = DEFAULT_WARMUP_DTYPES
) -> int:
    """Launch every variant (no guards). Returns the number of kernel launches."""
    from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

    launches = 0
    for dtype in dtypes:
        for grid, args, kwargs in build_warmup_launches(worker, dtype):
            _apply_grammar_bitmask_kernel[grid](*args, **kwargs)
            launches += 1
    return launches


def warmup_grammar_bitmask(worker: Any) -> int:
    """Pre-compile the grammar-bitmask variants. Never raises; returns launches."""
    if _is_disabled():
        logger.info_once("grammar-bitmask warmup skipped: %s is set", DISABLE_ENV)
        return 0
    if getattr(worker.device, "type", None) != "cuda" or not torch.cuda.is_available():
        return 0
    started = time.perf_counter()
    try:
        launches = run_warmup(worker)
        # Surface a compile/launch problem here instead of at first request.
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - warmup must never block worker startup
        logger.exception("grammar-bitmask warmup failed; continuing without it")
        return 0
    logger.info(
        "grammar-bitmask warmup: compiled %d variant(s) in %.2fs "
        "(dtypes=%s, vocab_size=%d, mask_stride=%d)",
        launches,
        time.perf_counter() - started,
        [str(d).removeprefix("torch.") for d in DEFAULT_WARMUP_DTYPES],
        int(worker.vocab_size),
        int(worker.mask_stride),
    )
    return launches
