"""Tests for store embedding configuration and helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dspy_lancedb_memory.store import LanceDSPyMemoryStore


class TestConstructorPaths:
    """Test constructor code paths for api_base/api_key handling."""

    def test_embed_calls_embedder(self, tmp_path):
        """Test _embed method calls embedder (line 123)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch("dspy_lancedb_memory.store.embeddings.dspy.LM", return_value=mock_lm),
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ),
        ):
            store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_embed",
                embedding_lm=mock_lm,
                embedding_dim=3,
                reranker=None,
            )
            # Call _embed to trigger line 123
            result = store._embed("test text")
            mock_embedder.assert_called_with(["test text"])
            assert result == [0.1, 0.2, 0.3]

    def test_embed_many_calls_embedder(self, tmp_path):
        """Test _embed_many method calls embedder (line 126)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with (
            patch("dspy_lancedb_memory.store.embeddings.dspy.LM", return_value=mock_lm),
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ),
        ):
            store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_embed_many",
                embedding_lm=mock_lm,
                embedding_dim=3,
                reranker=None,
            )
            # Call _embed_many to trigger line 126
            result = store._embed_many(["text1", "text2"])
            mock_embedder.assert_called_with(["text1", "text2"])
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_infer_embedding_dim_from_existing_table(self, tmp_path):
        """Test _infer_embedding_dim gets dimension from existing table (lines 130-133)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch("dspy_lancedb_memory.store.embeddings.dspy.LM", return_value=mock_lm),
            patch(
                "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
                return_value=mock_embedder,
            ),
        ):
            # First store creates the table with dimension 3
            _store1 = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_infer_dim",
                embedding_lm=mock_lm,
                embedding_dim=3,
                reranker=None,
            )

            # Second store should infer dimension from existing table
            store2 = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_infer_dim",
                embedding_lm=mock_lm,
                embedding_dim=None,  # Pass None to trigger _infer_embedding_dim
                reranker=None,
            )
            # Should infer dimension 3 from existing table
            assert store2.embedding_dim == 3

    def test_constructor_extracts_api_base_from_embedding_lm_kwargs(self, tmp_path):
        """Test constructor extracts api_base from embedding_lm kwargs."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch(
            "dspy_lancedb_memory.store.embeddings.dspy.Embedder",
            return_value=mock_embedder,
        ) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="huggingface/test-model",
                kwargs={"api_base": "http://custom.api"},
            )
            _store = LanceDSPyMemoryStore(
                uri=str(tmp_path),
                table_name="test_api_base",
                embedding_lm=embedding_lm,
                embedding_dim=3,
                reranker=None,
            )
            # Should rewrite model to "openai/" prefix when api_base is set
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            assert call_args[0][0] == "openai/test-model"
            assert call_args[1]["api_base"] == "http://custom.api"
            assert call_args[1]["encoding_format"] == "float"
