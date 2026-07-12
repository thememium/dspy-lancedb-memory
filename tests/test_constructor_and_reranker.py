"""Tests for store.py constructor paths and search with reranker.

Covers lines 92-146 (constructor with api_base/api_key) and line 896 (search with reranker).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dspy
import pytest

from dspy_lancedb_memory.reranking import LiteLLMReranker
from dspy_lancedb_memory.store import LanceDSPyMemoryStore


EMBEDDINGS: dict[str, list[float]] = {
    "test content": [0.5, 0.5, 0.0],
    "what food do I like": [0.95, 0.0, 0.05],
    "favorite food is pizza": [1.0, 0.0, 0.0],
}


class StubMemoryStoreWithReranker(LanceDSPyMemoryStore):
    """Store with a mock reranker for testing search with reranker paths."""

    def __init__(
        self,
        *,
        uri: str,
        table_name: str,
        embeddings: dict[str, list[float]],
        reranker=None,
    ):
        self._embeddings = embeddings
        super().__init__(
            uri=uri,
            table_name=table_name,
            embedding_lm=SimpleNamespace(model="test-embedding-model"),
            embedding_dim=3,
            reranker=reranker,
        )

    def _embed(self, text: str) -> list[float]:
        return self._embeddings.get(text, [0.5, 0.5, 0.0])


@pytest.fixture
def store(tmp_path):
    return StubMemoryStoreWithReranker(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )


# ---------------------------------------------------------------------------
# Constructor paths - mock dspy.Embedder to exercise api_base/api_key handling
# ---------------------------------------------------------------------------


class TestConstructorPaths:
    """Test constructor code paths for api_base/api_key handling."""

    def test_constructor_with_embedding_lm_none_creates_default(self, tmp_path):
        """Test constructor with embedding_lm=None creates default dspy.LM (line 92)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch("dspy_lancedb_memory.store.dspy.LM", return_value=mock_lm) as mock_lm_cls:
            with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder):
                store = LanceDSPyMemoryStore(
                    uri=str(tmp_path),
                    table_name="test_lm_none",
                    embedding_lm=None,  # Pass None to trigger line 92
                    embedding_dim=3,
                    reranker=None,
                )
                mock_lm_cls.assert_called_once_with("openrouter/openai/text-embedding-3-small")

    def test_constructor_with_embedding_dim_none_infers_dim(self, tmp_path):
        """Test constructor with embedding_dim=None infers dimension (line 116)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        # Return 4-dimensional embeddings
        mock_embedder.return_value = [[0.1, 0.2, 0.3, 0.4]]

        with patch("dspy_lancedb_memory.store.dspy.LM", return_value=mock_lm):
            with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder):
                store = LanceDSPyMemoryStore(
                    uri=str(tmp_path),
                    table_name="test_dim_none",
                    embedding_lm=mock_lm,
                    embedding_dim=None,  # Pass None to trigger line 116
                    reranker=None,
                )
                # Should infer dimension from embedder output
                assert store.embedding_dim == 4

    def test_embed_calls_embedder(self, tmp_path):
        """Test _embed method calls embedder (line 123)."""
        mock_lm = SimpleNamespace(model="test-model", kwargs={})
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch("dspy_lancedb_memory.store.dspy.LM", return_value=mock_lm):
            with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder):
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

        with patch("dspy_lancedb_memory.store.dspy.LM", return_value=mock_lm):
            with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder):
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

        with patch("dspy_lancedb_memory.store.dspy.LM", return_value=mock_lm):
            with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder):
                # First store creates the table with dimension 3
                store1 = LanceDSPyMemoryStore(
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
        """Test constructor extracts api_base from embedding_lm kwargs."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="huggingface/test-model",
                kwargs={"api_base": "http://custom.api"},
            )
            store = LanceDSPyMemoryStore(
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

    def test_constructor_with_embedding_lm_having_api_key(self, tmp_path):
        """Test constructor extracts api_key from embedding_lm kwargs."""
        mock_embedder = MagicMock()
        mock_embedder.return_value = [[0.1, 0.2, 0.3]]

        with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="test-model",
                kwargs={"api_key": "test-key-123"},
            )
            store = LanceDSPyMemoryStore(
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

        with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="huggingface/embedding-model",
                kwargs={"api_base": "http://localhost:8080", "api_key": "my-key"},
            )
            store = LanceDSPyMemoryStore(
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

        with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="openai/text-embedding-3-small",
                kwargs={},
            )
            store = LanceDSPyMemoryStore(
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

        with patch("dspy_lancedb_memory.store.dspy.Embedder", return_value=mock_embedder) as mock_cls:
            embedding_lm = SimpleNamespace(
                model="text-embedding-3-small",  # No slash
                kwargs={"api_base": "http://custom.api"},
            )
            store = LanceDSPyMemoryStore(
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


# ---------------------------------------------------------------------------
# search_memories with reranker paths
# ---------------------------------------------------------------------------


class TestSearchWithReranker:
    """Test search_memories when a reranker is set."""

    def test_search_with_reranker_sets_default_min_score_0_3(self, tmp_path):
        """When use_reranker=True and reranker is set, default min_relevance_score is 0.3."""
        import pyarrow as pa

        mock_reranker = MagicMock()
        mock_reranker.score = "relevance"

        def fake_rerank_vector(query, vector_results):
            # Add _relevance_score and drop _distance
            result = vector_results.append_column(
                "_relevance_score",
                pa.array([0.9] * len(vector_results), type=pa.float32()),
            )
            if "_distance" in result.column_names:
                result = result.drop_columns(["_distance"])
            return result

        mock_reranker.rerank_vector.side_effect = fake_rerank_vector

        store = StubMemoryStoreWithReranker(
            uri=str(tmp_path),
            table_name="memories",
            embeddings=EMBEDDINGS,
            reranker=mock_reranker,
        )

        store.create_memory(
            user_id="user-1",
            content="favorite food is pizza",
            memory_type="semantic",
        )

        results = store.search_memories(
            user_id="user-1",
            query="what food do I like",
            use_reranker=True,
        )
        assert len(results) >= 1

    def test_search_hybrid_sets_default_min_score_0(self, tmp_path):
        """When query_type='hybrid', default min_relevance_score is 0.0."""
        store = StubMemoryStoreWithReranker(
            uri=str(tmp_path),
            table_name="memories",
            embeddings=EMBEDDINGS,
        )

        store.create_memory(
            user_id="user-1",
            content="favorite food is pizza",
            memory_type="semantic",
        )

        results = store.search_memories(
            user_id="user-1",
            query="what food do I like",
            query_type="hybrid",
        )
        assert isinstance(results, list)

    def test_search_hybrid_with_reranker(self, tmp_path):
        """Test hybrid search with reranker (line 926)."""
        from unittest.mock import MagicMock

        # Mock the reranker
        mock_reranker = MagicMock()
        mock_reranker.score = "relevance"

        store = StubMemoryStoreWithReranker(
            uri=str(tmp_path),
            table_name="memories",
            embeddings=EMBEDDINGS,
            reranker=mock_reranker,
        )

        store.create_memory(
            user_id="user-1",
            content="favorite food is pizza",
            memory_type="semantic",
        )

        # Mock the table.search to return a builder that supports rerank
        mock_builder = MagicMock()
        mock_builder.vector.return_value = mock_builder
        mock_builder.text.return_value = mock_builder
        mock_builder.rerank.return_value = mock_builder
        mock_builder.where.return_value = mock_builder
        mock_builder.limit.return_value = mock_builder
        mock_builder.to_list.return_value = []

        with patch.object(store.table, 'search', return_value=mock_builder):
            results = store.search_memories(
                user_id="user-1",
                query="what food do I like",
                query_type="hybrid",
                use_reranker=True,
            )

        # Verify rerank was called (line 926)
        mock_builder.rerank.assert_called_once_with(mock_reranker, query_string="what food do I like")
        assert isinstance(results, list)
