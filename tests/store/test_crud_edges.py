"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from types import SimpleNamespace

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


def _active_rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__' AND is_active = true").to_list()


# ---------------------------------------------------------------------------
# list_memories edge cases
# ---------------------------------------------------------------------------


def test_list_memories_with_include_inactive(store):
    """Test list_memories with include_inactive=True."""
    memory = store.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=memory.id)

    active = store.list_memories(user_id="user-1")
    assert len(active) == 0

    all_memories = store.list_memories(user_id="user-1", include_inactive=True)
    assert len(all_memories) == 1


def test_list_memories_with_limit(store):
    """Test list_memories with limit."""
    for i in range(5):
        store.create_memory(
            user_id="user-1",
            content=f"content {i}",
            memory_type="semantic",
        )

    result = store.list_memories(user_id="user-1", limit=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# delete_memories_by_search edge cases
# ---------------------------------------------------------------------------


def test_delete_memories_by_search_no_candidates(store):
    """Test delete_memories_by_search returns empty when no candidates."""
    deleted = store.delete_memories_by_search(
        user_id="user-1",
        query="nonexistent memory",
    )

    assert deleted == []


def test_delete_memories_by_search_with_scope_and_metadata(store):
    """Test delete_memories_by_search with scope and metadata filter."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "a"},
        metadata={"source": "chat"},
    )
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "b"},
        metadata={"source": "doc"},
    )

    deleted = store.delete_memories_by_search(
        user_id="user-1",
        query="delete my pizza memory",
        scope={"tenant": "a"},
        metadata_filter={"source": "chat"},
        similarity_threshold=0.5,
    )

    assert len(deleted) == 1


def test_delete_missing_memory_is_noop(store):
    """Deleting an unknown id must not raise or write."""
    store.delete_memory(memory_id="missing-id")
    assert store.list_memories(user_id="user-1") == []


# ---------------------------------------------------------------------------
# update_memory_metadata edge cases
# ---------------------------------------------------------------------------


def test_update_memory_metadata_nonexistent(store):
    """Test update_memory_metadata with nonexistent ID."""
    import uuid

    result = store.update_memory_metadata(
        memory_id=str(uuid.uuid4()),
        metadata={"key": "value"},
    )
    assert result is None


def test_update_memory_metadata_replace(store):
    """Test update_memory_metadata with merge=False."""
    memory = store.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
        metadata={"source": "chat", "priority": "low"},
    )

    result = store.update_memory_metadata(
        memory_id=memory.id,
        metadata={"priority": "high"},
        merge=False,
    )

    assert result is not None
    assert result.metadata == {"priority": "high"}


# ---------------------------------------------------------------------------
# update_memory_scope edge cases
# ---------------------------------------------------------------------------


def test_update_memory_scope_nonexistent(store):
    """Test update_memory_scope with nonexistent ID."""
    import uuid

    result = store.update_memory_scope(
        memory_id=str(uuid.uuid4()),
        scope={"tenant": "a"},
    )
    assert result is None


def test_update_memory_scope_replace(store):
    """Test update_memory_scope with merge=False."""
    memory = store.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
        scope={"tenant": "a", "repo": "b"},
    )

    result = store.update_memory_scope(
        memory_id=memory.id,
        scope={"project": "c"},
        merge=False,
    )

    assert result is not None
    assert result.scope == {"project": "c"}


# ---------------------------------------------------------------------------
# create_memories error cases
# ---------------------------------------------------------------------------


def test_create_memories_extract_requires_contents(store):
    """Test create_memories with extract=True requires contents."""
    with pytest.raises(ValueError, match="contents is required when extract=True"):
        store.create_memories(
            user_id="user-1",
            contents=None,
            extract=True,
        )


def test_create_memories_extract_requires_nonempty_contents(store):
    """Test create_memories with extract=True requires non-empty contents."""
    with pytest.raises(ValueError, match="contents is required when extract=True"):
        store.create_memories(
            user_id="user-1",
            contents=[],
            extract=True,
        )


def test_create_memories_verbatim_requires_single_item(store):
    """Test create_memories with extract=False requires exactly one item."""
    with pytest.raises(ValueError, match="verbatim storage.*requires exactly one item"):
        store.create_memories(
            user_id="user-1",
            contents=[
                {"role": "user", "content": "item 1"},
                {"role": "user", "content": "item 2"},
            ],
            extract=False,
        )


def test_create_memories_verbatim_requires_contents(store):
    """Test create_memories with extract=False requires contents."""
    with pytest.raises(ValueError, match="verbatim storage.*requires exactly one item"):
        store.create_memories(
            user_id="user-1",
            contents=None,
            extract=False,
        )


def test_create_memories_verbatim_with_empty_list(store):
    """Test create_memories with extract=False requires non-empty contents."""
    with pytest.raises(ValueError, match="verbatim storage.*requires exactly one item"):
        store.create_memories(
            user_id="user-1",
            contents=[],
            extract=False,
        )


# ---------------------------------------------------------------------------
# get_memory_history with forward references
# ---------------------------------------------------------------------------


def test_get_memory_history_forward_references(store):
    """Test get_memory_history finds memories that reference the chain.

    Creates a chain: A <- B (B.replaces_id = A)
    Then creates C where C.replaces_id = B.
    Starting from A, the history should include all three.
    """
    # Create memory A
    mem_a = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    # Update A -> B (B.replaces_id = A)
    store.update_memory(memory_id=mem_a.id, content="hiking is my hobby")
    active_b = _active_rows(store)
    assert len(active_b) == 1

    # Now call get_memory_history starting from A
    # The changed loop should find B (which has replaces_id = A)
    history = store.get_memory_history(memory_id=mem_a.id)

    assert len(history) == 2
    contents = [h.content for h in history]
    assert "hiking is my hobby" in contents
    assert "I love hiking" in contents


def test_get_memory_history_forward_references_from_oldest(store):
    """Test get_memory_history starting from the oldest memory in chain."""
    # Create memory A
    mem_a = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    # Update A -> B (B.replaces_id = A)
    store.update_memory(memory_id=mem_a.id, content="hiking is my hobby")
    active_b = _active_rows(store)
    mem_b_id = active_b[0]["id"]

    # Update B -> C (C.replaces_id = B)
    store.update_memory(memory_id=mem_b_id, content="enjoys outdoor activities")

    # Start from A (oldest) - should find B and C via forward references
    history = store.get_memory_history(memory_id=mem_a.id)

    assert len(history) == 3
    contents = [h.content for h in history]
    assert "enjoys outdoor activities" in contents
    assert "hiking is my hobby" in contents
    assert "I love hiking" in contents


def test_get_memory_history_nonexistent_memory(store):
    """Test get_memory_history returns empty for nonexistent ID."""
    import uuid

    history = store.get_memory_history(memory_id=str(uuid.uuid4()))
    assert len(history) == 0


def test_get_memory_history_single_memory_no_chain(store):
    """Test get_memory_history for memory with no replacements."""
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    history = store.get_memory_history(memory_id=memory.id)

    assert len(history) == 1
    assert history[0].content == "favorite food is pizza"
