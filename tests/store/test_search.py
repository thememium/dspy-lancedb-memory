from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_lancedb_memory.store import LanceDSPyMemoryStore

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "their favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite food is pepperoni pizza": [0.95, 0.05, 0.0],
    "favorite color is blue": [0.99, 0.01, 0.0],
    "favorite color is red": [0.99, 0.01, 0.0],
    "favorite programming language is python": [0.0, 1.0, 0.0],
    "name is Edward": [0.9, 0.1, 0.0],
    "name is Edward Boswell": [0.88, 0.12, 0.0],
    "I love hiking": [0.0, 0.0, 1.0],
    "hiking is my hobby": [0.0, 0.0, 0.95],
    "enjoys outdoor activities": [0.0, 0.1, 0.9],
    "what are hobbies": [0.0, 0.05, 0.95],
    "what food do I like": [0.95, 0.0, 0.05],
    "car color is blue": [0.0, 0.5, 0.5],
    "car color is red": [0.0, 0.5, 0.5],
    # skip-threshold test vectors — L2 distance ≈ 0.057 with "I love hiking" (1 - dist ≈ 0.94)
    "hiking is enjoyable": [0.05, 0.02, 0.98],
    # skip-threshold test vectors — L2 distance ≈ 0.028 with "car color is blue" (1 - dist ≈ 0.97)
    "vehicle color blue": [0.02, 0.48, 0.5],
    # skip-threshold test vector — L2 distance ≈ 0.094 with "car color is blue" (1 - dist ≈ 0.91, >= 0.85 skip)
    "my car is blue": [0.08, 0.45, 0.5],
    "hiking": [0.0, 0.05, 0.95],
    # For reconciler test — similar to "name is Edward" but not identical
    "my name is edward": [0.85, 0.15, 0.0],
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
        return self._embeddings[text]


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> LanceDSPyMemoryStore:
    return StubMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )


def _rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__'").to_list()


def _active_rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__' AND is_active = true").to_list()


def _row_by_id(store: LanceDSPyMemoryStore, memory_id: str) -> dict:
    return next(row for row in _rows(store) if row["id"] == memory_id)


# ---------------------------------------------------------------------------
# Regression tests — capture CURRENT behavior before refactoring
# ---------------------------------------------------------------------------


def test_regression_search_memories_returns_results(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="hiking is my hobby",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
    )

    assert len(results) > 0
    result_contents = [r.content for r in results]
    assert any("hiking" in c for c in result_contents)


def test_regression_search_memories_respects_limit(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="hiking is my hobby",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="enjoys outdoor activities",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
        limit=1,
    )

    assert len(results) == 1


def test_regression_search_memories_respects_memory_type_filter(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        memory_type="preference",
    )

    assert len(results) >= 1
    for result in results:
        assert result.memory_type == "preference"


def test_search_memories_returns_empty_when_all_records_inactive(
    store: LanceDSPyMemoryStore,
) -> None:
    mem = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=mem.id)

    assert _active_rows(store) == []

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
    )
    assert results == []


# ---------------------------------------------------------------------------
# FTS index and hybrid search tests
# ---------------------------------------------------------------------------


def test_fts_index_is_created_on_table(store: LanceDSPyMemoryStore) -> None:
    """The FTS index on 'content' should exist after table creation."""
    indices = store.table.list_indices()
    fts_indices = [idx for idx in indices if idx.index_type == "FTS"]
    assert len(fts_indices) >= 1
    assert "content" in fts_indices[0].columns


def test_search_memories_refine_factor_does_not_error(
    store: LanceDSPyMemoryStore,
) -> None:
    """refine_factor should pass through to LanceDB without breaking the query."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        refine_factor=3,
    )

    assert len(results) >= 1


def test_search_memories_hybrid_returns_results(
    store: LanceDSPyMemoryStore,
) -> None:
    """Hybrid search (vector + FTS) should return results that match by keyword."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="hiking",
        query_type="hybrid",
        limit=5,
    )

    assert len(results) >= 1
    contents = [r.content for r in results]
    assert any("hiking" in c for c in contents)


def test_search_memories_hybrid_with_refine_factor(
    store: LanceDSPyMemoryStore,
) -> None:
    """Hybrid search with refine_factor should work without errors."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="hiking",
        query_type="hybrid",
        refine_factor=2,
        limit=5,
    )

    assert len(results) >= 1


def test_search_memories_vector_default_unchanged(
    store: LanceDSPyMemoryStore,
) -> None:
    """Default query_type='vector' should behave exactly as before."""
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
    )

    assert len(results) >= 1
    assert any("hiking" in r.content for r in results)
