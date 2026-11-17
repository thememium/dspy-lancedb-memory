"""Tests for dspy_lancedb_memory.memory — covering the memory factory module."""

from __future__ import annotations

import dspy
import pytest

from dspy_lancedb_memory import memory
from dspy_lancedb_memory.config import state


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config state before each test."""
    # Save original state
    orig_lm = state._lm
    orig_embedding_lm = state._embedding_lm
    orig_embedding_dim = state._embedding_dim
    orig_uri = state._uri
    orig_table_name = state._table_name
    orig_reranker_lm = state._reranker_lm
    orig_signature = state._signature
    orig_configured = state._configured

    yield

    # Restore original state
    state._lm = orig_lm
    state._embedding_lm = orig_embedding_lm
    state._embedding_dim = orig_embedding_dim
    state._uri = orig_uri
    state._table_name = orig_table_name
    state._reranker_lm = orig_reranker_lm
    state._signature = orig_signature
    state._configured = orig_configured


# ---------------------------------------------------------------------------
# memory.configure() wrapper
# ---------------------------------------------------------------------------


class TestMemoryConfigure:
    def test_configure_delegates_to_config(self):
        memory.configure(
            model="openrouter/openai/gpt-4o-mini",
            embedding_dim=1536,
            uri=".test_lancedb",
            table_name="test_table",
        )
        assert state._uri == ".test_lancedb"
        assert state._table_name == "test_table"
        assert state._embedding_dim == 1536
        assert state._lm is not None

    def test_configure_with_extraction_lm(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        memory.configure(extraction_lm=lm)
        assert state._lm is lm

    def test_configure_with_embedding_lm(self):
        lm = dspy.LM("openrouter/openai/text-embedding-3-small")
        memory.configure(embedding_lm=lm)
        assert state._embedding_lm is lm

    def test_configure_with_signature(self):
        class CustomSig(dspy.Signature):
            pass

        memory.configure(signature=CustomSig)
        assert state._signature is CustomSig

    def test_configure_with_reranker_lm(self):
        memory.configure(reranker_lm="cohere/rerank-english-v3.0")
        assert state._reranker_lm == "cohere/rerank-english-v3.0"
