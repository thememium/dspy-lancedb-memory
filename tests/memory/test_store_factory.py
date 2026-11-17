"""Tests for dspy_lancedb_memory.memory — covering the memory factory module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
# memory.Store() factory
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_store_with_explicit_params(self, tmp_path):
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]
        store = memory.Store(
            uri=str(tmp_path),
            table_name="test_memories",
            embedding_lm=SimpleNamespace(model="test-model"),
            embedding_dim=3,
            reranker=None,
        )
        assert store.table_name == "test_memories"
        assert store.embedding_dim == 3

    def test_store_uses_config_defaults(self, tmp_path):
        state._uri = str(tmp_path)
        state._table_name = "from_config"
        state._embedding_dim = 3
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]
        state._embedding_lm = SimpleNamespace(model="test-embedding")  # ty:ignore[invalid-assignment]

        store = memory.Store(reranker=None)
        assert store.table_name == "from_config"

    def test_store_explicit_overrides_config(self, tmp_path):
        state._uri = str(tmp_path)
        state._table_name = "from_config"
        state._embedding_dim = 3
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]
        state._embedding_lm = SimpleNamespace(model="test-embedding")  # ty:ignore[invalid-assignment]

        store = memory.Store(table_name="explicit", reranker=None)
        assert store.table_name == "explicit"

    def test_store_with_reranker_lm_string(self, tmp_path):
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]
        state._embedding_lm = SimpleNamespace(model="test-embedding")  # ty:ignore[invalid-assignment]
        state._embedding_dim = 3

        # This will try to create a LiteLLMReranker but won't actually call it
        store = memory.Store(
            uri=str(tmp_path),
            table_name="test",
            reranker_lm=None,  # No reranker
            reranker=None,
        )
        assert store.reranker is None

    def test_store_with_signature(self, tmp_path):
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]

        class CustomSig(dspy.Signature):
            pass

        store = memory.Store(
            uri=str(tmp_path),
            table_name="test",
            embedding_lm=SimpleNamespace(model="test-embedding"),
            embedding_dim=3,
            signature=CustomSig,
            reranker=None,
        )
        assert store._extraction_signature is CustomSig

    def test_store_with_config_signature(self, tmp_path):
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]

        class ConfigSig(dspy.Signature):
            pass

        state._signature = ConfigSig

        store = memory.Store(
            uri=str(tmp_path),
            table_name="test",
            embedding_lm=SimpleNamespace(model="test-embedding"),
            embedding_dim=3,
            reranker=None,
        )
        assert store._extraction_signature is ConfigSig

    def test_store_with_rerank_limit_multiplier(self, tmp_path):
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]

        store = memory.Store(
            uri=str(tmp_path),
            table_name="test",
            embedding_lm=SimpleNamespace(model="test-embedding"),
            embedding_dim=3,
            rerank_limit_multiplier=5,
            reranker=None,
        )
        assert store.rerank_limit_multiplier == 5

    def test_store_rerank_limit_multiplier_minimum_is_one(self, tmp_path):
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]

        store = memory.Store(
            uri=str(tmp_path),
            table_name="test",
            embedding_lm=SimpleNamespace(model="test-embedding"),
            embedding_dim=3,
            rerank_limit_multiplier=0,  # Should be clamped to 1
            reranker=None,
        )
        assert store.rerank_limit_multiplier == 1

    def test_store_with_reranker_lm_string_creates_default_reranker(self, tmp_path):
        """Test Store() with reranker_lm string creates LiteLLMReranker."""
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]

        with patch(
            "dspy_lancedb_memory.memory.reranker.LiteLLMReranker"
        ) as mock_reranker:
            mock_reranker.return_value = "mock-reranker"
            store = memory.Store(
                uri=str(tmp_path),
                table_name="test",
                embedding_lm=SimpleNamespace(model="test-embedding"),
                embedding_dim=3,
                reranker_lm="cohere/rerank-english-v3.0",
            )
            mock_reranker.assert_called_once_with(
                model="cohere/rerank-english-v3.0",
                column="content",
            )
            assert store.reranker == "mock-reranker"

    def test_store_with_reranker_lm_dspy_lm_with_api_base_and_key(self, tmp_path):
        """Test Store() with reranker_lm as dspy.LM with api_base and api_key."""
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]

        with patch(
            "dspy_lancedb_memory.memory.reranker.LiteLLMReranker"
        ) as mock_reranker:
            mock_reranker.return_value = "mock-reranker"
            # Create a dspy.LM with api_base and api_key
            reranker_lm = dspy.LM(
                "cohere/rerank-4-fast",
                api_base="http://custom.api",
                api_key="test-key",
            )
            store = memory.Store(
                uri=str(tmp_path),
                table_name="test",
                embedding_lm=SimpleNamespace(model="test-embedding"),
                embedding_dim=3,
                reranker_lm=reranker_lm,
            )
            # Should extract model string and api_base/api_key from dspy.LM
            mock_reranker.assert_called_once()
            call_kwargs = mock_reranker.call_args[1]
            assert call_kwargs["model"] == "cohere/rerank-4-fast"
            assert call_kwargs["column"] == "content"
            assert call_kwargs["api_base"] == "http://custom.api"
            assert call_kwargs["api_key"] == "test-key"
            assert store.reranker == "mock-reranker"

    def test_store_with_config_reranker_lm(self, tmp_path):
        """Test Store() uses config reranker_lm when not passed directly."""
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]
        state._reranker_lm = "cohere/rerank-english-v3.0"

        with patch(
            "dspy_lancedb_memory.memory.reranker.LiteLLMReranker"
        ) as mock_reranker:
            mock_reranker.return_value = "mock-reranker"
            store = memory.Store(
                uri=str(tmp_path),
                table_name="test",
                embedding_lm=SimpleNamespace(model="test-embedding"),
                embedding_dim=3,
            )
            mock_reranker.assert_called_once_with(
                model="cohere/rerank-english-v3.0",
                column="content",
            )
            assert store.reranker == "mock-reranker"

    def test_store_without_reranker_lm_returns_none(self, tmp_path):
        """Test Store() without reranker_lm returns None reranker."""
        state._lm = SimpleNamespace(model="test-model")  # ty:ignore[invalid-assignment]
        state._reranker_lm = None

        store = memory.Store(
            uri=str(tmp_path),
            table_name="test",
            embedding_lm=SimpleNamespace(model="test-embedding"),
            embedding_dim=3,
        )
        assert store.reranker is None
