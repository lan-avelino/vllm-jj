# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Azeus grammar-bitmask JIT warmup (deploy/jj-cand5).

No GPU compute is used — the kernel is replaced by a recorder — but importing
the worker package loads ``@triton.jit`` modules, so an active driver is
required; the module skips on CPU-only hosts (measured: 7 passed with a driver,
`TypeError: 'NoneType' object is not callable` from the Triton stub without
one). The variant matrix is documented in
``deploy/overlay-jj-cand5/WARMUP-MATRIX.md``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu import structured_outputs_warmup as warmup_mod

if not torch.cuda.is_available():  # pragma: no cover - GPU runners only
    pytest.skip(
        "needs an active driver: the import chain loads @triton.jit modules",
        allow_module_level=True,
    )

# DeepSeek-V4.1-like: divisible by 16, while cdiv(vocab, 32) is not.
VOCAB_SIZE = 129280
MASK_STRIDE = 8
MAX_NUM_LOGITS = 4


class _Recorder:
    """Stands in for ``_apply_grammar_bitmask_kernel``."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple, dict]] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.fixture
def worker() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        vocab_size=VOCAB_SIZE,
        mask_stride=MASK_STRIDE,
        logits_indices=torch.zeros(MAX_NUM_LOGITS, dtype=torch.int32),
        grammar_bitmask=torch.zeros(
            (MAX_NUM_LOGITS, cdiv(VOCAB_SIZE, 32)), dtype=torch.int32
        ),
    )


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.structured_outputs._apply_grammar_bitmask_kernel", rec
    )
    return rec


def test_matrix_covers_both_input_pointer_variants(worker):
    launches = warmup_mod.build_warmup_launches(worker, torch.bfloat16)
    assert len(launches) == 2

    (_, none_args, _), (_, tensor_args, _) = launches
    # Variant 1: the common path passes None for both input pointers.
    assert none_args[4] is None and none_args[5] is None
    # Variant 2: the invalid-draft path passes real tensors, with the dtypes the
    # production call site uses (input_ids int32, input_batch.logits_indices int64).
    assert isinstance(tensor_args[4], torch.Tensor)
    assert tensor_args[4].dtype == torch.int32
    assert isinstance(tensor_args[5], torch.Tensor)
    assert tensor_args[5].dtype == torch.int64


def test_specialization_classes_match_production(worker):
    (grid, args, kwargs), _ = warmup_mod.build_warmup_launches(worker, torch.bfloat16)
    from vllm.v1.worker.gpu.structured_outputs import GRAMMAR_BITMASK_BLOCK_SIZE

    logits, logits_stride, logits_indices, cu_num_logits, _, _, bitmask = args[:7]

    # Constexprs must equal what the real launch passes, or we warm a different
    # compile key than production uses (silent, expensive failure).
    assert kwargs["BLOCK_SIZE"] == GRAMMAR_BITMASK_BLOCK_SIZE
    assert GRAMMAR_BITMASK_BLOCK_SIZE == 8192
    assert kwargs["MASK_STRIDE"] == MASK_STRIDE

    # bf16 logits, row stride == vocab_size (divisible by 16 -> aligned class).
    assert logits.dtype == torch.bfloat16
    assert logits_stride == VOCAB_SIZE
    assert logits_stride % 16 == 0

    # Scalars/pointers the kernel reads.
    assert cu_num_logits.dtype == torch.int32
    assert bitmask.dtype == torch.int32
    assert bitmask.stride(0) == cdiv(VOCAB_SIZE, 32)
    # bitmask_stride is the one *generic* (non div-16) scalar class.
    assert bitmask.stride(0) % 16 != 0
    assert grid == (
        warmup_mod.WARMUP_ROWS,
        cdiv(VOCAB_SIZE, GRAMMAR_BITMASK_BLOCK_SIZE),
    )

    # Reuse the worker's own buffers so alignment/strides match production.
    assert logits_indices.data_ptr() == worker.logits_indices.data_ptr()


def test_dtype_axis_is_covered(worker):
    for dtype in (torch.bfloat16, torch.float16):
        (_, args, _), _ = warmup_mod.build_warmup_launches(worker, dtype)
        assert args[0].dtype == dtype


def test_invalid_draft_store_is_inert_and_in_bounds(worker):
    """The warmup executes the kernel, so its dummies must be memory-safe."""
    _, (_, tensor_args, _) = warmup_mod.build_warmup_launches(worker, torch.bfloat16)
    logits_indices, input_ids, input_logits_indices = (
        tensor_args[2],
        tensor_args[4],
        tensor_args[5],
    )
    # Non-negative mapping values => `invalid_draft` is False => no store.
    assert torch.all(logits_indices >= 0)
    # ...and even if it did store, the row it targets is inside input_ids.
    assert input_logits_indices.numel() > 0
    assert int(input_logits_indices.max()) < input_ids.numel()


def test_run_warmup_launches_every_variant(worker, recorder):
    launches = warmup_mod.run_warmup(worker)
    assert launches == 2 * len(warmup_mod.DEFAULT_WARMUP_DTYPES)
    assert len(recorder.calls) == launches
    dtypes = [call[1][0].dtype for call in recorder.calls]
    assert dtypes == [torch.bfloat16, torch.bfloat16, torch.float16, torch.float16]


def test_guard_skips_off_cuda(worker):
    assert warmup_mod.warmup_grammar_bitmask(worker) == 0


def test_escape_hatch_skips_warmup(monkeypatch):
    monkeypatch.setenv(warmup_mod.DISABLE_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_device = SimpleNamespace(type="cuda")
    worker = SimpleNamespace(
        device=fake_device,
        vocab_size=VOCAB_SIZE,
        mask_stride=MASK_STRIDE,
        logits_indices=torch.zeros(1, dtype=torch.int32),
        grammar_bitmask=torch.zeros((1, cdiv(VOCAB_SIZE, 32)), dtype=torch.int32),
    )
    assert warmup_mod.warmup_grammar_bitmask(worker) == 0
