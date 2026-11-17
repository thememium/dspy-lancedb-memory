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


def test_regression_create_memory_increases_row_count(
    store: LanceDSPyMemoryStore,
) -> None:
    assert len(_rows(store)) == 0

    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    assert len(_rows(store)) == 1


def test_regression_create_memory_returns_correct_fields(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    assert memory.user_id == "user-1"
    assert memory.content == "favorite food is pizza"
    assert memory.memory_type == "semantic"
    assert memory.id
    assert memory.created_at
    assert memory.updated_at


def test_regression_delete_memory_deactivates_row(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1

    store.delete_memory(memory_id=memory.id)

    # Soft delete: row stays but is_active flips to False
    assert len(_rows(store)) == 1
    row = _row_by_id(store, memory.id)
    assert row["is_active"] is False


def test_regression_delete_memory_sets_inactive(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    store.delete_memory(memory_id=memory.id)

    row = _row_by_id(store, memory.id)
    assert row is not None
    assert row["is_active"] is False


def test_create_memory_defaults(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC7: create_memory() returns Memory with is_active=True and
    replaces_id=None."""
    memory = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    assert memory.is_active is True
    assert memory.replaces_id is None


def test_update_memory_nonexistent_id_is_noop(
    store: LanceDSPyMemoryStore,
) -> None:
    import uuid

    before = len(_rows(store))
    store.update_memory(
        memory_id=str(uuid.uuid4()),
        content="favorite food is pizza",
    )
    assert len(_rows(store)) == before


def test_update_memory_on_inactive_record_is_noop(
    store: LanceDSPyMemoryStore,
) -> None:
    mem = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=mem.id)
    rows_after_delete = len(_rows(store))
    assert rows_after_delete == 1

    store.update_memory(
        memory_id=mem.id,
        content="favorite food is pepperoni pizza",
    )
    assert len(_rows(store)) == rows_after_delete


def test_delete_memory_on_already_inactive_is_idempotent(
    store: LanceDSPyMemoryStore,
) -> None:
    mem = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=mem.id)

    store.delete_memory(memory_id=mem.id)

    assert len(_rows(store)) == 1
    row = _row_by_id(store, mem.id)
    assert row["is_active"] is False


def test_list_and_get_memory_with_filters(store: LanceDSPyMemoryStore) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata={"source": "doc"},
        scope={"tenant": "tenant-a"},
    )
    store.create_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
        metadata={"source": "chat"},
        scope={"tenant": "tenant-b"},
    )

    listed = store.list_memories(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
        metadata_filter={"source": "doc"},
    )
    fetched = store.get_memory(memory_id=memory.id)

    assert [r.id for r in listed] == [memory.id]
    assert fetched is not None
    assert fetched.id == memory.id


def test_update_metadata_and_scope_create_history(store: LanceDSPyMemoryStore) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata={"source": "chat"},
        scope={"tenant": "tenant-a"},
    )

    with_metadata = store.update_memory_metadata(
        memory_id=memory.id,
        metadata={"priority": "high"},
    )
    assert with_metadata is not None
    with_scope = store.update_memory_scope(
        memory_id=with_metadata.id,
        scope={"region": "us-central"},
    )

    assert with_scope is not None
    assert with_scope.metadata == {"source": "chat", "priority": "high"}
    assert with_scope.scope == {"tenant": "tenant-a", "region": "us-central"}

    history = store.get_memory_history(memory_id=with_scope.id)
    assert [row.content for row in history] == [
        "favorite food is pizza",
        "favorite food is pizza",
        "favorite food is pizza",
    ]
    assert history[0].id == with_scope.id
    assert history[-1].id == memory.id


# ---------------------------------------------------------------------------
# Group 2: list_memories with limit + scope/metadata
# ---------------------------------------------------------------------------


def test_list_memories_limit_with_scope_filter(
    store: LanceDSPyMemoryStore,
) -> None:
    for i in range(5):
        store.create_memory(
            user_id="user-1",
            content="favorite food is pizza",
            memory_type="semantic",
            scope={"group": "a", "index": str(i)},
        )
    for i in range(5):
        store.create_memory(
            user_id="user-1",
            content="favorite color is blue",
            memory_type="semantic",
            scope={"group": "b", "index": str(i)},
        )

    results = store.list_memories(
        user_id="user-1",
        scope={"group": "a"},
        limit=3,
    )

    assert len(results) == 3
    for r in results:
        assert r.scope["group"] == "a"


def test_list_memories_limit_with_metadata_filter(
    store: LanceDSPyMemoryStore,
) -> None:
    for i in range(4):
        store.create_memory(
            user_id="user-1",
            content="favorite food is pizza",
            memory_type="semantic",
            metadata={"source": "doc", "index": str(i)},
        )
    store.create_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
        metadata={"source": "chat"},
    )

    results = store.list_memories(
        user_id="user-1",
        metadata_filter={"source": "doc"},
        limit=2,
    )

    assert len(results) == 2
    for r in results:
        assert r.metadata["source"] == "doc"


# ---------------------------------------------------------------------------
# Group 3: get_memory with include_inactive
# ---------------------------------------------------------------------------


def test_get_memory_returns_active_by_default(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    fetched = store.get_memory(memory_id=memory.id)
    assert fetched is not None
    assert fetched.id == memory.id

    store.delete_memory(memory_id=memory.id)

    gone = store.get_memory(memory_id=memory.id)
    assert gone is None


def test_get_memory_include_inactive_returns_deleted(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    store.delete_memory(memory_id=memory.id)

    active = store.get_memory(memory_id=memory.id)
    assert active is None

    inactive = store.get_memory(memory_id=memory.id, include_inactive=True)
    assert inactive is not None
    assert inactive.id == memory.id
    assert inactive.is_active is False


# ---------------------------------------------------------------------------
# Group 4: get_memory_history edge cases
# ---------------------------------------------------------------------------


def test_get_memory_history_single_memory(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    history = store.get_memory_history(memory_id=memory.id)

    assert len(history) == 1
    assert history[0].id == memory.id
    assert history[0].content == "favorite food is pizza"


def test_get_memory_history_two_hops(
    store: LanceDSPyMemoryStore,
) -> None:
    mem_a = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    store.update_memory(
        memory_id=mem_a.id,
        content="hiking is my hobby",
    )
    active_rows = _active_rows(store)
    assert len(active_rows) == 1
    mem_b = active_rows[0]

    history = store.get_memory_history(memory_id=mem_b["id"])

    assert len(history) == 2
    assert history[0].id == mem_b["id"]
    assert history[0].content == "hiking is my hobby"
    assert history[1].id == mem_a.id
    assert history[1].content == "I love hiking"
