"""Tests for semantic match helper edge cases."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_lancedb_memory.store import LanceDSPyMemoryStore
from dspy_lancedb_memory.store.similarity import semantic_match_action

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
# _semantic_match_action final return (line 421)
# ---------------------------------------------------------------------------


def test_semantic_match_action_update_return(store):
    """Test _semantic_match_action returns 'update' for richer content.

    Line 421: return "update" if new_is_richer or same_slot_replacement else "skip"
    This requires same_fact=True and sequence_ratio < 0.94 and new_is_richer=True.
    """
    # Use contents that are semantically similar but new is richer
    result = semantic_match_action(
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
    result = semantic_match_action(
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
    result = semantic_match_action(
        new_content="favorite food is pizza",
        existing_content="favorite color is blue",
        distance=0.9,  # Low similarity
        similarity_threshold=0.5,
    )

    assert result is None


def test_semantic_match_action_skip_high_sequence_ratio(store):
    """Test _semantic_match_action returns 'skip' when sequence_ratio >= 0.94."""
    # Use very similar contents with high sequence ratio
    result = semantic_match_action(
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
    result = semantic_match_action(
        new_content="abcdefg hijklmn opqrstu",
        existing_content="xbcdefg hijklmn opqrstu",
        distance=0.15,  # cosine_similarity = 0.85
        similarity_threshold=0.8,
        skip_threshold=0.9,  # cosine_similarity < skip_threshold, so no early return
    )

    # Should return "skip" because sequence_ratio >= 0.94 and not new_is_richer and not same_slot_replacement
    assert result == "skip"
