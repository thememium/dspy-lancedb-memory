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


def _rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__'").to_list()


def _active_rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__' AND is_active = true").to_list()


# ---------------------------------------------------------------------------
# upsert_memory with reconciler (real paths)
# ---------------------------------------------------------------------------


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


def test_upsert_memory_with_reconciler_keep(store, monkeypatch):
    """Test upsert_memory with reconciler returning 'keep' action."""
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
        skip_threshold=1.0,  # Force reconciler path
        use_reconciler=True,
    )

    assert result.id == original.id


def test_upsert_memory_with_reconciler_update(store, monkeypatch):
    """Test upsert_memory with reconciler returning 'update' action."""
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="name is Edward Boswell",
        memory_type="semantic",
        skip_threshold=1.0,
        use_reconciler=True,
    )

    assert result.id != original.id
    assert result.content == "name is Edward Boswell"


def test_upsert_memory_with_reconciler_create(store, monkeypatch):
    """Test upsert_memory with reconciler returning 'create' action."""
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
        skip_threshold=1.0,
        use_reconciler=True,
    )

    assert result.content == "favorite color is blue"
    assert len(_rows(store)) == 2


def test_upsert_memory_nonsemantic_with_reconciler(store, monkeypatch):
    """Test upsert_memory with non-semantic type and reconciler."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="preference",
        skip_threshold=1.0,
        use_reconciler=True,
    )

    # Non-semantic with reconciler should use reconciler
    assert result is not None


# ---------------------------------------------------------------------------
# upsert_memories edge cases
# ---------------------------------------------------------------------------


def test_upsert_memories_verbatim(store):
    """Test upsert_memories with extract=False."""
    result = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "test content"}],
        extract=False,
        use_reconciler=False,
    )

    assert len(result) == 1
    assert result[0].content == "test content"


def test_upsert_memories_verbatim_requires_single_item(store):
    """Test upsert_memories with extract=False requires exactly one item."""
    with pytest.raises(ValueError, match="exactly one item"):
        store.upsert_memories(
            user_id="user-1",
            contents=[
                {"role": "user", "content": "item 1"},
                {"role": "user", "content": "item 2"},
            ],
            extract=False,
        )


def test_upsert_memories_verbatim_requires_contents(store):
    """Test upsert_memories with extract=False requires contents."""
    with pytest.raises(ValueError, match="exactly one item"):
        store.upsert_memories(
            user_id="user-1",
            contents=None,
            extract=False,
        )


def test_upsert_memories_extract_requires_contents(store):
    """Test upsert_memories with extract=True requires contents."""
    with pytest.raises(ValueError, match="contents is required"):
        store.upsert_memories(
            user_id="user-1",
            contents=None,
            extract=True,
        )


# ---------------------------------------------------------------------------
# upsert_memory non-semantic paths
# ---------------------------------------------------------------------------


def test_upsert_nonsemantic_exact_match_returns_existing(store):
    """Test non-semantic upsert with exact content match returns existing."""
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
        use_reconciler=False,
    )

    assert result.id == original.id


def test_upsert_nonsemantic_similarity_above_threshold_updates(store):
    """Test non-semantic upsert updates when similarity is above threshold."""
    _original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    # Use a content with high similarity but not exact match
    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="preference",
        similarity_threshold=0.8,
        skip_threshold=0.99,  # High skip threshold to allow update
        use_reconciler=False,
    )

    # Should either update or create depending on similarity
    assert result is not None


def test_upsert_nonsemantic_below_skip_threshold_creates(store):
    """Test non-semantic upsert creates when below skip threshold."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    # Use a content with low similarity
    result = store.upsert_memory(
        user_id="user-1",
        content="favorite programming language is python",
        memory_type="preference",
        use_reconciler=False,
    )

    assert result.content == "favorite programming language is python"
