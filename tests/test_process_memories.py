from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest

from dspy_lancedb_memory.models import MemoryOperation
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
    "delete my pizza memory": [1.0, 0.0, 0.0],
    "remove my preference for pizza": [1.0, 0.0, 0.0],
    "forget that I like pizza": [1.0, 0.0, 0.0],
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


def _rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__' AND is_active = true").to_list()


def test_delete_memories_by_search_removes_matching_memories(store):
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    deleted = store.delete_memories_by_search(
        user_id="user-1",
        query="delete my pizza memory",
        similarity_threshold=0.5,
    )

    assert len(deleted) == 1
    assert deleted[0].content == "favorite food is pizza"
    assert len(_rows(store)) == 0


def test_delete_memories_by_search_respects_similarity_threshold(store):
    store.create_memory(
        user_id="user-1",
        content="favorite programming language is python",
        memory_type="semantic",
    )

    deleted = store.delete_memories_by_search(
        user_id="user-1",
        query="favorite food is pizza",
        similarity_threshold=0.9,
    )

    assert len(deleted) == 0
    assert len(_rows(store)) == 1


def test_process_memories_updates_when_extract_returns_update(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
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

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell"}],
        extract=True,
        similarity_threshold=0.5,
        use_reconciler=False,
    )

    assert len(created) == 1
    assert created[0].content == "name is Edward Boswell"
    assert len(deleted) == 0
    assert len(_rows(store)) == 1


def test_process_memories_deletes_when_extract_returns_delete(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    search_query="pizza memory",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Delete my pizza memory"}],
        extract=True,
        similarity_threshold=0.5,
    )

    assert len(created) == 0
    assert len(deleted) == 1
    assert deleted[0].content == "favorite food is pizza"
    assert len(_rows(store)) == 0


def test_process_memories_creates_when_extract_returns_create(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="create",
                    content="favorite color is blue",
                    memory_type="semantic",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My favorite color is blue"}],
        extract=True,
    )

    assert len(created) == 1
    assert created[0].content == "favorite color is blue"
    assert len(deleted) == 0
    assert len(_rows(store)) == 1


def test_process_memories_verbatim_non_extract_creates(store):
    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[
            {"role": "user", "content": "favorite programming language is python"}
        ],
        extract=False,
    )

    assert len(created) == 1
    assert created[0].content == "favorite programming language is python"
    assert len(deleted) == 0
    assert len(_rows(store)) == 1


def test_process_memories_skips_invalid_actions(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="invalid",
                    content="some content",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Some message"}],
        extract=True,
    )

    assert len(created) == 0
    assert len(deleted) == 0
    assert len(_rows(store)) == 0


def test_process_memories_delete_with_exact_match(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    content="favorite food is pizza",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Delete favorite food is pizza"}],
        extract=True,
    )

    assert len(created) == 0
    assert len(deleted) == 1
    assert deleted[0].content == "favorite food is pizza"
    assert len(_rows(store)) == 0


def test_process_memories_multiple_operations(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="old fact to remove",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    search_query="old fact",
                ),
                MemoryOperation(
                    action="create",
                    content="new fact to remember",
                    memory_type="semantic",
                ),
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[
            {"role": "user", "content": "Delete the old fact and remember the new fact"}
        ],
        extract=True,
        similarity_threshold=0.5,
    )

    assert len(created) == 1
    assert created[0].content == "new fact to remember"
    assert len(deleted) == 1
    assert deleted[0].content == "old fact to remove"
    assert len(_rows(store)) == 1
