"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from dspy_lancedb_memory.store import LanceDSPyMemoryStore
from dspy_lancedb_memory.store.table import get_vector_dim

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
# _get_vector_dim edge cases
# ---------------------------------------------------------------------------


class TestGetVectorDim:
    def test_variable_size_list_returns_none(self):
        table = pa.table(
            {"vector": pa.array([[0.1, 0.2]], type=pa.list_(pa.float32()))}
        )

        assert get_vector_dim(table) is None

    def test_table_without_vector_column_returns_none(self):
        table = pa.table({"id": [1]})

        assert get_vector_dim(table) is None


# ---------------------------------------------------------------------------
# _get_or_create_table edge cases
# ---------------------------------------------------------------------------


def test_get_or_create_table_with_dimension_mismatch(tmp_path):
    """Test table recreation when vector dimension doesn't match."""
    # Create a store with dimension 3
    store1 = StubMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )
    store1.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
    )

    # Create another store with different dimension - should recreate table
    different_embeddings = {"test": [1.0, 0.0]}
    store2 = LanceDSPyMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embedding_lm=SimpleNamespace(model="test-model"),
        embedding_dim=2,
        reranker=None,
    )
    # Override _embed to use our test embeddings
    store2._embed = lambda text: different_embeddings.get(text, [0.5, 0.5])
    store2.table = store2._get_or_create_table()

    # Table should have been recreated with new dimension
    assert store2.embedding_dim == 2


# ---------------------------------------------------------------------------
# _ensure_fts_index exception handling (lines 490-491)
# ---------------------------------------------------------------------------


def test_ensure_fts_index_handles_exception(store):
    """Test _ensure_fts_index handles exceptions gracefully (lines 490-491)."""
    from unittest.mock import MagicMock

    from lancedb.index import FTS

    # Create a mock table with no indexes that raises on create_index
    mock_table = MagicMock()
    mock_table.list_indices.return_value = []
    mock_table.create_index.side_effect = Exception("FTS index error")

    # Call _ensure_fts_index - should not raise, just log
    store._ensure_fts_index(mock_table)

    # Verify the modern index API was used
    mock_table.create_index.assert_called_once()
    assert mock_table.create_index.call_args[0][0] == "content"
    assert isinstance(mock_table.create_index.call_args[1]["config"], FTS)


def test_ensure_fts_index_creates_index(store):
    """Test _ensure_fts_index creates FTS index successfully."""
    from unittest.mock import MagicMock

    from lancedb.index import FTS

    # Create a mock table with no indexes
    mock_table = MagicMock()
    mock_table.list_indices.return_value = []
    mock_table.create_index.return_value = None

    # Call _ensure_fts_index - should succeed
    store._ensure_fts_index(mock_table)

    # Verify the modern index API was used
    mock_table.create_index.assert_called_once()
    assert mock_table.create_index.call_args[0][0] == "content"
    assert isinstance(mock_table.create_index.call_args[1]["config"], FTS)


def test_ensure_fts_index_skips_existing_index(store):
    """An existing content index means no rebuild on startup."""
    from unittest.mock import MagicMock

    mock_table = MagicMock()
    index = MagicMock()
    index.columns = ["content"]
    mock_table.list_indices.return_value = [index]

    store._ensure_fts_index(mock_table)

    mock_table.create_index.assert_not_called()
