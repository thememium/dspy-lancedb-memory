"""Tests for reconciliation and memory-history edge cases."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_lancedb_memory.models import PendingReconciliation, ReconciledMemory
from dspy_lancedb_memory.store import LanceDSPyMemoryStore

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite food is pepperoni pizza": [0.95, 0.05, 0.0],
    "I love hiking": [0.0, 0.0, 1.0],
    "hiking is my hobby": [0.0, 0.0, 0.95],
    "enjoys outdoor activities": [0.0, 0.1, 0.9],
    "name is Edward": [0.9, 0.1, 0.0],
    "name is Edward Boswell": [0.88, 0.12, 0.0],
    "favorite color is blue": [0.99, 0.01, 0.0],
    "car color is blue": [0.0, 0.5, 0.5],
    "test content": [0.5, 0.5, 0.0],
    # For _semantic_match_action testing - vectors that produce specific distances
    "my name is edward": [0.85, 0.15, 0.0],  # Similar to "name is Edward"
    "the name is edward": [0.87, 0.13, 0.0],  # Similar with prefix overlap
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
def store(tmp_path):
    return StubMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )


# ---------------------------------------------------------------------------
# _apply_reconciliations with "keep" and no existing_row (lines 1130-1136)
# ---------------------------------------------------------------------------


def test_apply_reconciliations_keep_without_existing_row(store):
    """Test _apply_reconciliations when keep action has no existing_row.

    This covers lines 1130-1136: the else branch that queries the table
    by memory_id when existing_row is not set.
    """
    # Create a memory to keep
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    # Create a PendingReconciliation with "keep" action but NO existing_row
    decision = ReconciledMemory(
        action="keep",
        memory_id=memory.id,
        final_content="favorite food is pizza",
        final_type="semantic",
    )
    pending = PendingReconciliation(
        content="favorite food is pizza",
        inferred_type="semantic",
        decision=decision,
        user_id="user-1",
        session_id="",
        conversation_id="",
        # existing_row is NOT set (defaults to None)
    )

    results = store._apply_reconciliations([pending])

    assert len(results) == 1
    assert results[0].content == "favorite food is pizza"
    assert results[0].id == memory.id


def test_apply_reconciliations_keep_with_existing_row(store):
    """Test _apply_reconciliations when keep action HAS existing_row."""
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    decision = ReconciledMemory(
        action="keep",
        memory_id=memory.id,
        final_content="favorite food is pizza",
        final_type="semantic",
    )
    pending = PendingReconciliation(
        content="favorite food is pizza",
        inferred_type="semantic",
        decision=decision,
        user_id="user-1",
        session_id="",
        conversation_id="",
        existing_row=memory,  # existing_row IS set
    )

    results = store._apply_reconciliations([pending])

    assert len(results) == 1
    assert results[0].content == "favorite food is pizza"


def test_apply_reconciliations_keep_nonexistent_id(store):
    """Test _apply_reconciliations when keep action references nonexistent ID."""
    import uuid

    decision = ReconciledMemory(
        action="keep",
        memory_id=str(uuid.uuid4()),
        final_content="test",
        final_type="semantic",
    )
    pending = PendingReconciliation(
        content="test",
        inferred_type="semantic",
        decision=decision,
        user_id="user-1",
        session_id="",
        conversation_id="",
    )

    results = store._apply_reconciliations([pending])

    # Should return empty since the memory_id doesn't exist
    assert len(results) == 0


def test_apply_reconciliations_update_action(store):
    """Test _apply_reconciliations with update action."""
    memory = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    decision = ReconciledMemory(
        action="update",
        memory_id=memory.id,
        final_content="name is Edward Boswell",
        final_type="semantic",
    )
    pending = PendingReconciliation(
        content="name is Edward Boswell",
        inferred_type="semantic",
        decision=decision,
        user_id="user-1",
        session_id="",
        conversation_id="",
    )

    results = store._apply_reconciliations([pending])

    assert len(results) == 1
    assert results[0].content == "name is Edward Boswell"


def test_apply_reconciliations_create_action(store):
    """Test _apply_reconciliations with create action."""
    decision = ReconciledMemory(
        action="create",
        memory_id="",
        final_content="new memory content",
        final_type="semantic",
    )
    pending = PendingReconciliation(
        content="new memory content",
        inferred_type="semantic",
        decision=decision,
        user_id="user-1",
        session_id="session-1",
        conversation_id="conv-1",
    )

    results = store._apply_reconciliations([pending])

    assert len(results) == 1
    assert results[0].content == "new memory content"
    assert results[0].session_id == "session-1"


def test_apply_reconciliations_mixed_actions(store):
    """Test _apply_reconciliations with multiple actions."""
    existing = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    pending = [
        PendingReconciliation(
            content="name is Edward",
            inferred_type="semantic",
            decision=ReconciledMemory(
                action="keep",
                memory_id=existing.id,
                final_content="name is Edward",
                final_type="semantic",
            ),
            user_id="user-1",
            session_id="",
            conversation_id="",
        ),
        PendingReconciliation(
            content="favorite color is blue",
            inferred_type="semantic",
            decision=ReconciledMemory(
                action="create",
                memory_id="",
                final_content="favorite color is blue",
                final_type="semantic",
            ),
            user_id="user-1",
            session_id="",
            conversation_id="",
        ),
    ]

    results = store._apply_reconciliations(pending)

    assert len(results) == 2
    contents = {r.content for r in results}
    assert "name is Edward" in contents
    assert "favorite color is blue" in contents
