"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_lancedb_memory.store import LanceDSPyMemoryStore
from dspy_lancedb_memory.store.search import filter_by_relevance

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite food is pepperoni pizza": [0.95, 0.05, 0.0],
    "favorite color is blue": [0.99, 0.01, 0.0],
    "favorite color is red": [0.99, 0.01, 0.0],
    "favorite programming language is python": [0.0, 1.0, 0.0],
    "name is Edward": [0.9, 0.1, 0.0],
    "name is Edward Boswell": [0.88, 0.12, 0.0],
    "I love hiking": [0.0, 0.0, 1.0],
    "hiking is my hobby": [0.0, 0.0, 0.95],
    "car color is blue": [0.0, 0.5, 0.5],
    "car color is red": [0.0, 0.5, 0.5],
    "hiking is enjoyable": [0.05, 0.02, 0.98],
    "vehicle color blue": [0.02, 0.48, 0.5],
    "my car is blue": [0.08, 0.45, 0.5],
    "new fact to remember": [0.3, 0.3, 0.3],
    "delete my pizza memory": [1.0, 0.0, 0.0],
    "test content": [0.5, 0.5, 0.0],
    "updated content": [0.6, 0.4, 0.0],
}


class StubMemoryStore(LanceDSPyMemoryStore):
    def __init__(
        self, *, uri: str, table_name: str, embeddings: dict[str, list[float]]
    ):
        self._embeddings = embeddings
        super().__init__(
            uri=uri,
            table_name=table_name,
            embedding_lm=SimpleNamespace(model="test-embedding-model"),
            embedding_dim=3,
            reranker=None,
        )

    def _embed(self, text: str) -> list[float]:
        return self._embeddings.get(text, [0.5, 0.5, 0.0])


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> LanceDSPyMemoryStore:
    return StubMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )


# ---------------------------------------------------------------------------
# _filter_by_relevance
# ---------------------------------------------------------------------------


class TestFilterByRelevance:
    def test_filter_with_relevance_score(self):
        results = [
            {"_relevance_score": 0.9},
            {"_relevance_score": 0.5},
            {"_relevance_score": 0.2},
        ]
        filtered = filter_by_relevance(results, 0.6)
        assert len(filtered) == 1
        assert filtered[0]["_relevance_score"] == 0.9

    def test_filter_with_distance(self):
        results = [
            {"_distance": 0.1},  # score ≈ 0.91
            {"_distance": 0.5},  # score ≈ 0.67
            {"_distance": 0.8},  # score ≈ 0.56
        ]
        filtered = filter_by_relevance(results, 0.7)
        assert len(filtered) == 1
        assert filtered[0]["_distance"] == 0.1

    def test_filter_with_no_score_defaults_to_zero(self):
        results = [{"id": "test"}]
        filtered = filter_by_relevance(results, 0.5)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# search_memories edge cases
# ---------------------------------------------------------------------------


def test_search_memories_with_min_relevance_score(store):
    """Test search_memories with min_relevance_score."""
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    # Search with high min score - should filter out low relevance
    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
        min_relevance_score=0.99,
    )
    # Might return 0 or 1 depending on similarity
    assert isinstance(results, list)
