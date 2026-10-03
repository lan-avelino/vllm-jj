# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the json_schema whitespace runaway (Azeus deployment patch).

With xgrammar 0.2.6 the JSON grammar allows unbounded ``[ \\n\\r\\t]*`` between
tokens. When a model's plan conflicts with the schema (it wants ``true`` where
the schema demands ``"yes"``/``"no"``), it can emit ``\\r\\n`` until
max_tokens and return unparseable JSON. ``VLLM_XGRAMMAR_MAX_WHITESPACE_CNT``
bounds every whitespace run, so the model is forced back onto the schema.
"""

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257
SCHEMA = (
    '{"type": "object", "additionalProperties": false, '
    '"required": ["answer"], '
    '"properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}}'
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def backend(tokenizer) -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=VOCAB_SIZE)


def _single_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text)
    assert len(ids) == 1, f"{text!r} is not a single gpt2 token: {ids}"
    return ids[0]


def _start_answer(backend: XgrammarBackend, tokenizer):
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA)
    assert grammar.accept_tokens("req", tokenizer.encode('{"answer":'))
    return grammar


def _crlf_tokens(tokenizer, chars: int) -> list[int]:
    cr, lf = _single_token(tokenizer, "\r"), _single_token(tokenizer, "\n")
    return [cr if i % 2 == 0 else lf for i in range(chars)]


def test_whitespace_run_is_capped_by_default(backend, tokenizer, monkeypatch):
    monkeypatch.delenv("VLLM_XGRAMMAR_MAX_WHITESPACE_CNT", raising=False)
    run = _crlf_tokens(tokenizer, 33)
    grammar = _start_answer(backend, tokenizer)

    for token in run[:32]:
        assert grammar.accept_tokens("req", [token])
    assert not grammar.accept_tokens("req", [run[32]])

    assert grammar.accept_tokens("req", tokenizer.encode('"yes"}'))


def test_zero_disables_the_cap(backend, tokenizer, monkeypatch):
    monkeypatch.setenv("VLLM_XGRAMMAR_MAX_WHITESPACE_CNT", "0")
    grammar = _start_answer(backend, tokenizer)

    for token in _crlf_tokens(tokenizer, 400):
        assert grammar.accept_tokens("req", [token])
