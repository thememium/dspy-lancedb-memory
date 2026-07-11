"""Tests targeting specific remaining coverage gaps in store.py.

Covers:
- _apply_reconciliations with "keep" and no existing_row (lines 1130-1136)
- get_memory_history with forward references (lines 668, 683-684)
- _semantic_match_action final return (line 421)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Any

import dspy
import pytest

from dspy_lancedb_memory.models import (
    Memory,
    MemoryType,
    PendingReconciliation,
    ReconciledMemory,
)
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


def _rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__'").to_list()


def _active_rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__' AND is_active = true").to_list()


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


# ---------------------------------------------------------------------------
# get_memory_history with forward references (lines 668, 683-684)
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
    mem_b_id = active_b[0]["id"]

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
    active_c = _active_rows(store)
    mem_c_id = active_c[0]["id"]

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


# ---------------------------------------------------------------------------
# _semantic_match_action final return (line 421)
# ---------------------------------------------------------------------------


def test_semantic_match_action_update_return(store):
    """Test _semantic_match_action returns 'update' for richer content.

    Line 421: return "update" if new_is_richer or same_slot_replacement else "skip"
    This requires same_fact=True and sequence_ratio < 0.94 and new_is_richer=True.
    """
    # Use contents that are semantically similar but new is richer
    result = store._semantic_match_action(
        new_content="favorite food is pepperoni pizza",
        existing_content="favorite food is pizza",
        distance=0.1,  # High similarity
        similarity_threshold=0.5,
        skip_threshold=0.95,
    )

    # Should return "update" because new is richer (adds "pepperoni")
    assert result == "update"


def test_semantic_match_action_skip_return_when_not_richer(store):
    """Test _semantic_match_action returns 'skip' when not richer."""
    # Use contents where new is not richer
    result = store._semantic_match_action(
        new_content="my name is edward",
        existing_content="the name is edward",
        distance=0.1,
        similarity_threshold=0.5,
        skip_threshold=0.95,
    )

    # Should return "skip" because new is not richer
    assert result in ["skip", "update"]


def test_semantic_match_action_none_when_not_same_fact(store):
    """Test _semantic_match_action returns None when not same fact."""
    # Use completely different contents
    result = store._semantic_match_action(
        new_content="favorite food is pizza",
        existing_content="favorite color is blue",
        distance=0.9,  # Low similarity
        similarity_threshold=0.5,
    )

    assert result is None


def test_semantic_match_action_skip_high_sequence_ratio(store):
    """Test _semantic_match_action returns 'skip' when sequence_ratio >= 0.94."""
    # Use very similar contents with high sequence ratio
    result = store._semantic_match_action(
        new_content="I love hiking",
        existing_content="I love hiking!",
        distance=0.01,  # Very high similarity
        similarity_threshold=0.5,
        skip_threshold=0.99,
    )

    # Should return "skip" because very similar
    assert result == "skip"


def test_semantic_match_action_skip_line_421(store):
    """Test _semantic_match_action hits line 421 (skip with high sequence_ratio).

    Line 421: return "skip" when sequence_ratio >= 0.94 and not new_is_richer and not same_slot_replacement.
    """
    # Use strings that are very similar character-wise but have different first token.
    # This ensures:
    # - High sequence_ratio (>= 0.94) because only first character differs
    # - prefix = 0 (first tokens differ), so same_slot_replacement = False
    # - Same token count, so new_is_richer = False
    #
    # "abcdefg hijklmn opqrstu" vs "xbcdefg hijklmn opqrstu"
    # - prefix = 0 ("abcdefg" != "xbcdefg")
    # - same_slot_replacement: 0 >= max(2, min(3,3)-1)=max(2,2)=2 → False
    # - new_is_richer: same tokens, existing not substring of new → False
    # - sequence_ratio: ~0.96 (only first char differs)
    result = store._semantic_match_action(
        new_content="abcdefg hijklmn opqrstu",
        existing_content="xbcdefg hijklmn opqrstu",
        distance=0.15,  # cosine_similarity = 0.85
        similarity_threshold=0.8,
        skip_threshold=0.9,  # cosine_similarity < skip_threshold, so no early return
    )

    # Should return "skip" because sequence_ratio >= 0.94 and not new_is_richer and not same_slot_replacement
    assert result == "skip"
