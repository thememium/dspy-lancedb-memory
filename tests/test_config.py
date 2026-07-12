"""Tests for dspy_lancedb_memory.config — covering all uncovered paths."""

from __future__ import annotations

import dspy
import pytest

from dspy_lancedb_memory import config


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config state before each test."""
    # Save original state
    orig_lm = config._lm
    orig_embedding_lm = config._embedding_lm
    orig_embedding_dim = config._embedding_dim
    orig_uri = config._uri
    orig_table_name = config._table_name
    orig_reranker_lm = config._reranker_lm
    orig_signature = config._signature
    orig_configured = config._configured

    yield

    # Restore original state
    config._lm = orig_lm
    config._embedding_lm = orig_embedding_lm
    config._embedding_dim = orig_embedding_dim
    config._uri = orig_uri
    config._table_name = orig_table_name
    config._reranker_lm = orig_reranker_lm
    config._signature = orig_signature
    config._configured = orig_configured


# ---------------------------------------------------------------------------
# configure() function
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_configure_sets_lm_from_model_string(self):
        result = config.configure(model="openrouter/openai/gpt-4o-mini")
        assert result is not None
        assert config._lm is result

    def test_configure_sets_extraction_lm_directly(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        result = config.configure(extraction_lm=lm)
        assert result is lm
        assert config._lm is lm

    def test_configure_extraction_lm_takes_precedence_over_model(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        result = config.configure(model="openrouter/openai/gpt-4o", extraction_lm=lm)
        assert result is lm

    def test_configure_sets_embedding_lm(self):
        lm = dspy.LM("openrouter/openai/text-embedding-3-small")
        config.configure(embedding_lm=lm)
        assert config._embedding_lm is lm

    def test_configure_sets_embedding_dim(self):
        config.configure(embedding_dim=768)
        assert config._embedding_dim == 768

    def test_configure_sets_uri(self):
        config.configure(uri=".custom_lancedb")
        assert config._uri == ".custom_lancedb"

    def test_configure_sets_table_name(self):
        config.configure(table_name="custom_memories")
        assert config._table_name == "custom_memories"

    def test_configure_sets_signature(self):
        class CustomSig(dspy.Signature):
            pass

        config.configure(signature=CustomSig)
        assert config._signature is CustomSig

    def test_configure_sets_reranker_lm(self):
        config.configure(reranker_lm="cohere/rerank-english-v3.0")
        assert config._reranker_lm == "cohere/rerank-english-v3.0"

    def test_configure_sets_configured_flag(self):
        assert config._configured is False
        config.configure()
        assert config._configured is True

    def test_configure_returns_none_without_lm(self):
        result = config.configure(uri=".test")
        assert result is None


# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------


class TestGetters:
    def test_get_configured_lm_raises_when_not_set(self):
        config._lm = None
        with pytest.raises(RuntimeError, match="No language model configured"):
            config.get_configured_lm()

    def test_get_configured_lm_returns_set_lm(self):
        lm = dspy.LM("openrouter/openai/gpt-4o-mini")
        config._lm = lm
        assert config.get_configured_lm() is lm

    def test_get_configured_returns_flag(self):
        config._configured = False
        assert config.get_configured() is False
        config._configured = True
        assert config.get_configured() is True

    def test_get_embedding_config_returns_tuple(self):
        lm = dspy.LM("openrouter/openai/text-embedding-3-small")
        config._embedding_lm = lm
        config._embedding_dim = 1536
        result = config.get_embedding_config()
        assert result == (lm, 1536)

    def test_get_store_config_returns_tuple(self):
        config._uri = ".test_lancedb"
        config._table_name = "test_table"
        result = config.get_store_config()
        assert result == (".test_lancedb", "test_table")

    def test_get_reranker_lm_config(self):
        config._reranker_lm = "cohere/rerank-english-v3.0"
        assert config.get_reranker_lm_config() == "cohere/rerank-english-v3.0"

    def test_get_reranker_lm_config_none(self):
        config._reranker_lm = None
        assert config.get_reranker_lm_config() is None

    def test_get_signature_config(self):
        config._signature = None
        assert config.get_signature_config() is None

        class CustomSig(dspy.Signature):
            pass

        config._signature = CustomSig
        assert config.get_signature_config() is CustomSig
