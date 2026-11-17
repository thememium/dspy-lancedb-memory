"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest

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


# ---------------------------------------------------------------------------
# process_memories edge cases
# ---------------------------------------------------------------------------


def test_process_memories_requires_contents(store):
    """Test process_memories requires contents."""
    with pytest.raises(ValueError, match="contents is required"):
        store.process_memories(
            user_id="user-1",
            contents=None,
        )


def test_process_memories_delete_with_search_query(store, monkeypatch):
    """Test process_memories delete action with search_query."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.process.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                {
                    "action": "delete",
                    "content": "",
                    "search_query": "pizza",
                    "memory_type": "",
                }
            ]
        ),
    )

    # This should use delete_memories_by_search
    # But we need to handle the fact that the mock returns dict not MemoryOperation
    # Let me adjust this test
    from dspy_lancedb_memory.models import MemoryOperation

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.process.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    content="",
                    search_query="pizza",
                    memory_type="",
                )
            ]
        ),
    )

    _created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Delete pizza"}],
        extract=True,
        similarity_threshold=0.5,
    )

    assert len(deleted) >= 1


# ---------------------------------------------------------------------------
# process_memories edge cases for uncovered paths
# ---------------------------------------------------------------------------


def test_process_memories_update_action(store, monkeypatch):
    """Test process_memories update action (line 1486)."""
    from dspy_lancedb_memory.models import MemoryOperation

    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.process.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="update",
                    content="name is Edward Boswell",
                    memory_type="semantic",
                )
            ]
        ),
    )

    created, _deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell"}],
        extract=True,
        use_reconciler=False,
    )

    assert len(created) == 1
    assert created[0].content == "name is Edward Boswell"


def test_process_memories_empty_search_query_skips(store, monkeypatch):
    """Test process_memories skips delete with empty search query."""
    from dspy_lancedb_memory.models import MemoryOperation

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.process.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    content="",
                    search_query="",
                    memory_type="",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Delete something"}],
        extract=True,
    )

    assert len(created) == 0
    assert len(deleted) == 0
