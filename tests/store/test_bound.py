"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest

from dspy_lancedb_memory.models import ReconciledMemory
from dspy_lancedb_memory.store import LanceDSPyMemoryStore

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


class StubReconciler:
    """Deterministic reconciler for testing."""

    def __call__(self, *, new_memory_content, new_memory_type, existing_memories):
        for existing in existing_memories:
            if existing["content"] == new_memory_content:
                return dspy.Prediction(
                    reconciled=ReconciledMemory(
                        action="keep",
                        memory_id=existing["id"],
                        final_content=existing["content"],
                        final_type=existing["type"],
                    )
                )
            if new_memory_content.startswith(existing["content"] + " "):
                return dspy.Prediction(
                    reconciled=ReconciledMemory(
                        action="update",
                        memory_id=existing["id"],
                        final_content=new_memory_content,
                        final_type=existing["type"],
                    )
                )
        return dspy.Prediction(
            reconciled=ReconciledMemory(
                action="create",
                memory_id="",
                final_content=new_memory_content,
                final_type=new_memory_type,
            )
        )


# ---------------------------------------------------------------------------
# BoundMemoryStore methods
# ---------------------------------------------------------------------------


def test_bound_store_process_memories(store):
    """Test BoundMemoryStore.process_memories."""
    bound = store.with_scope(
        user_id="user-1",
        session_id="session-1",
        scope={"tenant": "tenant-a"},
    )

    created, _deleted = bound.process_memories(
        contents=[{"role": "user", "content": "I love pizza"}],
        extract=False,
    )

    assert len(created) == 1
    assert created[0].session_id == "session-1"
    assert created[0].scope == {"tenant": "tenant-a"}


def test_bound_store_delete_memories_by_search(store):
    """Test BoundMemoryStore.delete_memories_by_search."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    bound.create_memory(
        content="favorite food is pizza",
        memory_type="semantic",
    )

    deleted = bound.delete_memories_by_search(
        query="delete my pizza memory",
        similarity_threshold=0.5,
    )

    assert len(deleted) == 1
    assert deleted[0].content == "favorite food is pizza"


def test_bound_store_upsert_memories(store, monkeypatch):
    """Test BoundMemoryStore.upsert_memories."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite food is pizza", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    results = bound.upsert_memories(
        contents=[{"role": "user", "content": "I like pizza"}],
        extract=True,
        num_threads=1,
    )

    assert len(results) == 1
    assert results[0].scope == {"tenant": "tenant-a"}


def test_bound_store_scope_merging(store):
    """Test BoundMemoryStore._scope merges scopes correctly."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "a", "repo": "b"},
    )

    # Test internal _scope method
    merged = bound._scope({"project": "c"})
    assert merged == {"tenant": "a", "repo": "b", "project": "c"}


def test_bound_store_create_memories_extract(store, monkeypatch):
    """Test BoundMemoryStore.create_memories with extract=True."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.read.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite food is pizza", "semantic")]
        ),
    )

    results = bound.create_memories(
        contents=[{"role": "user", "content": "I like pizza"}],
        extract=True,
    )

    assert len(results) == 1
    assert results[0].scope == {"tenant": "tenant-a"}
