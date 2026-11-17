"""Tests for store constructor paths."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dspy_lancedb_memory.store import LanceDSPyMemoryStore


class TestConstructorPaths:
    """Test constructor code paths for api_base/api_key handling."""

    def test_constructor_with_embedding_lm_none_creates_default(self, tmp_path):
        """Test constructor with embedding_lm=None creates default dspy.LM (line 92)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.LM", return_value=mock_lm
            ) as mock_lm_cls,
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ),
        ):
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_lm_none",
                embedding_lm=None,  # Pass None to trigger line 92
                embedding_dim=3,
                reranker=None,
            )
            mock_lm_cls.assert_called_once_with(
                "openrouter/openai/text-embedding-3-small"
            )

    def test_constructor_with_embedding_dim_none_infers_dim(self, tmp_path):
        """Test constructor with embedding_dim=None infers dimension (line 116)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        # Return 4-dimensional embeddings
        mock_embedder.return_value = [[0.1, 0.2, 0.3, 0.4]]

        with (
            patch("dspy_lancedb_memory.store.embeddings.dspy.LM", return_value=mock_lm),
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ),
        ):
            store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_dim_none",
                embedding_lm=mock_lm,
                embedding_dim=None,  # Pass None to trigger line 116
                reranker=None,
            )
            # Should infer dimension from embedder output
            assert store.embedding_dim == 4

    def test_constructor_with_embedding_lm_having_api_key(self, tmp_path):
        """Test constructor extracts api_key from embedding_lm kwargs."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch(
            "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
            return_value=mock_embedder,
        ) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="test-model",
                kwargs={"api_key": "test-key-123"},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_api_key",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            assert call_args[1]["api_key"] == "test-key-123"
            # No api_base set, so no encoding_format and no model rewrite
            assert "encoding_format" not in call_args[1]

    def test_constructor_with_embedding_lm_having_both_api_base_and_key(self, tmp_path):
        """Test constructor extracts both api_base and api_key."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch(
            "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
            return_value=mock_embedder,
        ) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="huggingface/embedding-model",
                kwargs={"api_base": "http://localhost:8080", "api_key": "my-key"},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_both",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            assert call_args[0][0] == "openai/embedding-model"
            assert call_args[1]["api_base"] == "http://localhost:8080"
            assert call_args[1]["api_key"] == "my-key"
            assert call_args[1]["encoding_format"] == "float"

    def test_constructor_with_no_api_base_no_model_rewrite(self, tmp_path):
        """Test constructor does NOT rewrite model when api_base is not set."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch(
            "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
            return_value=mock_embedder,
        ) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="openai/text-embedding-3-small",
                kwargs={},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_no_rewrite",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            # No api_base, so model should NOT be rewritten
            assert call_args[0][0] == "openai/text-embedding-3-small"
            assert "encoding_format" not in call_args[1]

    def test_constructor_with_api_base_but_no_slash_in_model(self, tmp_path):
        """Test constructor does NOT rewrite model when model has no '/' prefix."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch(
            "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
            return_value=mock_embedder,
        ) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="text-embedding-3-small",  # No slash
                kwargs={"api_base": "http://custom.api"},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_no_slash",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            # Model has no "/" so no rewrite should happen
            assert call_args[0][0] == "text-embedding-3-small"
            assert call_args[1]["api_base"] == "http://custom.api"
            # encoding_format is NOT set because model has no "/"
            assert "encoding_format" not in call_args[1]

    def test_constructor_openrouter_embedding_uses_openrouter_endpoint(self, tmp_path):
        """OpenRouter embedding models route through an OpenAI-compatible client."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ) as mock_cls,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}),
        ):
            embedding_lm = SimpleNamespace(
                model="openrouter/openai/text-embedding-3-small",
                kwargs={},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_openrouter",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )

        call_args = mock_cls.call_args
        # Embedder strips the first segment, leaving OpenRouter's full id.
        assert call_args[0][0] == "openai/openai/text-embedding-3-small"
        assert call_args[1]["api_base"] == "https://openrouter.ai/api/v1"
        assert call_args[1]["api_key"] == "sk-or-test"
        assert call_args[1]["encoding_format"] == "float"

    def test_constructor_openrouter_embedding_wraps_non_openai_model(self, tmp_path):
        """Non-OpenAI OpenRouter model ids get the OpenAI-compatible prefix."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ) as mock_cls,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}),
        ):
            embedding_lm = SimpleNamespace(
                model="openrouter/cohere/embed-v4.0",
                kwargs={},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_openrouter_cohere",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )

        assert mock_cls.call_args[0][0] == "openai/cohere/embed-v4.0"

    def test_constructor_openrouter_embedding_without_key(self, tmp_path):
        """Missing OPENROUTER_API_KEY leaves api_key unset for the SDK default."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ) as mock_cls,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}),
        ):
            embedding_lm = SimpleNamespace(
                model="openrouter/openai/text-embedding-3-small",
                kwargs={},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_openrouter_no_key",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )

        assert "api_key" not in mock_cls.call_args[1]

    def test_constructor_openrouter_embedding_respects_explicit_endpoint(
        self, tmp_path
    ):
        """Explicit api_base/api_key override the OpenRouter defaults."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ) as mock_cls,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}),
        ):
            embedding_lm = SimpleNamespace(
                model="openrouter/openai/text-embedding-3-small",
                kwargs={
                    "api_base": "https://gateway.example.com/v1",
                    "api_key": "gateway-key",
                },
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_openrouter_override",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )

        assert mock_cls.call_args[0][0] == "openai/openai/text-embedding-3-small"
        assert mock_cls.call_args[1]["api_base"] == "https://gateway.example.com/v1"
        assert mock_cls.call_args[1]["api_key"] == "gateway-key"
