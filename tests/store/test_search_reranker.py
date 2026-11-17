"""Tests for search_memories with reranker paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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

    def test_hybrid_reranker_strips_row_id(self, tmp_path):
        """Rows returned by the reranker must not leak the internal _rowid."""
        mock_reranker = MagicMock()
        mock_reranker.score = "relevance"
        mock_reranker.rerank_hybrid.return_value.to_pylist.return_value = [
            {"id": "x", "_rowid": "x", "content": "favorite food is pizza"}
        ]

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

        rows = store._hybrid_results(
            query="what food do I like",
            filters=["user_id = 'user-1'"],
            fetch_limit=5,
            refine_factor=None,
            use_reranker=True,
        )

        assert rows
        assert "_rowid" not in rows[0]

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

        mock_reranker.rerank_hybrid.return_value.to_pylist.return_value = []

        results = store.search_memories(
            user_id="user-1",
            query="what food do I like",
            query_type="hybrid",
            use_reranker=True,
        )

        # Hybrid search merges local vector/FTS results through the reranker.
        mock_reranker.rerank_hybrid.assert_called_once()
        assert mock_reranker.rerank_hybrid.call_args[0][0] == "what food do I like"
        assert isinstance(results, list)
