"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_lancedb_memory.models import MemoryType, memory_type_value
from dspy_lancedb_memory.store import LanceDSPyMemoryStore
from dspy_lancedb_memory.store.similarity import semantic_match_action

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
# _semantic_match_action edge cases
# ---------------------------------------------------------------------------


def test_semantic_match_action_empty_content(store):
    """Test _semantic_match_action with empty normalized content."""
    result = semantic_match_action(
        new_content="!!!",
        existing_content="hello",
        distance=0.5,
        similarity_threshold=0.85,
    )
    # Empty normalized content returns None
    assert result is None


def test_semantic_match_action_identical_normalized(store):
    """Test _semantic_match_action with identical normalized content."""
    result = semantic_match_action(
        new_content="Hello World!",
        existing_content="hello world",
        distance=0.1,
        similarity_threshold=0.85,
    )
    assert result == "skip"


# ---------------------------------------------------------------------------
# MemoryType value edge cases
# ---------------------------------------------------------------------------


def test_memory_type_value_with_string(store):
    """Test _memory_type_value with string input."""
    result = memory_type_value("semantic")
    assert result == "semantic"


def test_memory_type_value_with_enum(store):
    """Test _memory_type_value with enum input."""
    result = memory_type_value(MemoryType.PREFERENCE)
    assert result == "preference"


def test_memory_type_value_with_none(store):
    """Test _memory_type_value with None input."""
    result = memory_type_value(None)
    assert result is None
