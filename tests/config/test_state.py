"""Tests for dspy_lancedb_memory.config — covering all uncovered paths."""

from __future__ import annotations

import dspy
import pytest

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
# configure() function
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_configure_sets_lm_from_model_string(self):
        result = state.configure(model="openrouter/openai/gpt-4o-mini")
        assert result is not None
        assert state._lm is result

    def test_configure_sets_extraction_lm_directly(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        result = state.configure(extraction_lm=lm)
        assert result is lm
        assert state._lm is lm

    def test_configure_extraction_lm_takes_precedence_over_model(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        result = state.configure(model="openrouter/openai/gpt-4o", extraction_lm=lm)
        assert result is lm

    def test_configure_sets_embedding_lm(self):
        lm = dspy.LM("openrouter/openai/text-embedding-3-small")
        state.configure(embedding_lm=lm)
        assert state._embedding_lm is lm

    def test_configure_sets_embedding_dim(self):
        state.configure(embedding_dim=768)
        assert state._embedding_dim == 768

    def test_configure_sets_uri(self):
        state.configure(uri=".custom_lancedb")
        assert state._uri == ".custom_lancedb"

    def test_configure_sets_table_name(self):
        state.configure(table_name="custom_memories")
        assert state._table_name == "custom_memories"

    def test_configure_sets_signature(self):
        class CustomSig(dspy.Signature):
            pass

        state.configure(signature=CustomSig)
        assert state._signature is CustomSig

    def test_configure_sets_reranker_lm(self):
        state.configure(reranker_lm="cohere/rerank-english-v3.0")
        assert state._reranker_lm == "cohere/rerank-english-v3.0"

    def test_configure_sets_configured_flag(self):
        assert state._configured is False
        state.configure()
        assert state._configured is True

    def test_configure_returns_none_without_lm(self):
        result = state.configure(uri=".test")
        assert result is None


# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------


class TestGetters:
    def test_get_configured_lm_raises_when_not_set(self):
        state._lm = None
        with pytest.raises(RuntimeError, match="No language model configured"):
            state.get_configured_lm()

    def test_get_configured_lm_returns_set_lm(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        state._lm = lm
        assert state.get_configured_lm() is lm

    def test_get_configured_returns_flag(self):
        state._configured = False
        assert state.get_configured() is False
        state._configured = True
        assert state.get_configured() is True

    def test_get_embedding_config_returns_tuple(self):
        lm = dspy.LM("openrouter/openai/text-embedding-3-small")
        state._embedding_lm = lm
        state._embedding_dim = 1536
        result = state.get_embedding_config()
        assert result == (lm, 1536)

    def test_get_store_config_returns_tuple(self):
        state._uri = ".test_lancedb"
        state._table_name = "test_table"
        result = state.get_store_config()
        assert result == (".test_lancedb", "test_table")

    def test_get_reranker_lm_config(self):
        state._reranker_lm = "cohere/rerank-english-v3.0"
        assert state.get_reranker_lm_config() == "cohere/rerank-english-v3.0"

    def test_get_reranker_lm_config_none(self):
        state._reranker_lm = None
        assert state.get_reranker_lm_config() is None

    def test_get_signature_config(self):
        state._signature = None
        assert state.get_signature_config() is None

        class CustomSig(dspy.Signature):
            pass

        state._signature = CustomSig
        assert state.get_signature_config() is CustomSig
