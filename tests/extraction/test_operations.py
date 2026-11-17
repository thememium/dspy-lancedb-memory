"""Tests for dspy_lancedb_memory.extraction — covering forward methods."""

from __future__ import annotations

import dspy

from dspy_lancedb_memory.extraction import MemoryOperationExtractor
from dspy_lancedb_memory.models import MemoryOperation

# ---------------------------------------------------------------------------
# MemoryOperationExtractor.forward()
# ---------------------------------------------------------------------------


class TestMemoryOperationExtractor:
    def test_forward_cleans_valid_operations(self):
        extractor = MemoryOperationExtractor()

        # Mock the inner ChainOfThought to return raw operations
        def mock_extract(messages):
            return dspy.Prediction(
                operations=[
                    MemoryOperation(action="CREATE", content="test"),
                    MemoryOperation(action="  update  ", content="test2"),
                    MemoryOperation(action="delete", content="test3"),
                ]
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        ops = result.operations
        assert len(ops) == 3
        assert ops[0].action == "create"
        assert ops[1].action == "update"
        assert ops[2].action == "delete"

    def test_forward_filters_invalid_actions(self):
        extractor = MemoryOperationExtractor()

        def mock_extract(messages):
            return dspy.Prediction(
                operations=[
                    MemoryOperation(action="create", content="test"),
                    MemoryOperation(action="invalid_action", content="test2"),
                    MemoryOperation(action="", content="test3"),
                ]
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        ops = result.operations
        assert len(ops) == 1
        assert ops[0].action == "create"

    def test_forward_handles_non_list_operations(self):
        extractor = MemoryOperationExtractor()

        def mock_extract(messages):
            return dspy.Prediction(
                operations=MemoryOperation(action="create", content="test")
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        ops = result.operations
        assert len(ops) == 1
        assert ops[0].action == "create"

    def test_forward_handles_empty_operations(self):
        extractor = MemoryOperationExtractor()

        def mock_extract(messages):
            return dspy.Prediction(operations=None)

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        ops = result.operations
        assert len(ops) == 0

    def test_forward_handles_empty_list(self):
        extractor = MemoryOperationExtractor()

        def mock_extract(messages):
            return dspy.Prediction(operations=[])

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        ops = result.operations
        assert len(ops) == 0
