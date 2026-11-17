"""Tests for dspy_lancedb_memory.extraction — covering forward methods."""

from __future__ import annotations

import dspy

from dspy_lancedb_memory.extraction import MemoryExtractor
from dspy_lancedb_memory.models import MemoryItem

# ---------------------------------------------------------------------------
# MemoryExtractor.forward()
# ---------------------------------------------------------------------------


class TestMemoryExtractor:
    def test_forward_cleans_valid_memories(self):
        extractor = MemoryExtractor()

        def mock_extract(messages):
            return dspy.Prediction(
                memories=[
                    MemoryItem(content="  test memory  ", type="semantic"),
                    MemoryItem(content="another memory", type="preference"),
                ]
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        memories = result.memories
        assert len(memories) == 2
        assert memories[0][0] == "test memory"  # stripped
        assert memories[0][1].value == "semantic"
        assert memories[1][0] == "another memory"
        assert memories[1][1].value == "preference"

    def test_forward_handles_non_list_items(self):
        extractor = MemoryExtractor()

        def mock_extract(messages):
            return dspy.Prediction(
                memories=MemoryItem(content="single item", type="semantic")
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        memories = result.memories
        assert len(memories) == 1
        assert memories[0][0] == "single item"

    def test_forward_filters_empty_content(self):
        extractor = MemoryExtractor()

        def mock_extract(messages):
            return dspy.Prediction(
                memories=[
                    MemoryItem(content="valid", type="semantic"),
                    MemoryItem(content="", type="semantic"),
                    MemoryItem(content="  ", type="semantic"),
                ]
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        memories = result.memories
        assert len(memories) == 1
        assert memories[0][0] == "valid"

    def test_forward_preserves_metadata(self):
        extractor = MemoryExtractor()

        def mock_extract(messages):
            return dspy.Prediction(
                memories=[
                    MemoryItem(
                        content="test",
                        type="semantic",
                        metadata={"source": "chat"},
                    )
                ]
            )

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        memories = result.memories
        assert len(memories) == 1
        assert memories[0][2] == {"source": "chat"}

    def test_forward_handles_missing_metadata(self):
        extractor = MemoryExtractor()

        def mock_extract(messages):
            # Simulate an item without metadata attribute
            item = MemoryItem(content="test", type="semantic")
            delattr(item, "metadata")
            return dspy.Prediction(memories=[item])

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        memories = result.memories
        assert len(memories) == 1
        assert memories[0][2] == {}

    def test_forward_handles_none_metadata(self):
        extractor = MemoryExtractor()

        def mock_extract(messages):
            # Create an item and then set metadata to None via object.__setattr__
            item = MemoryItem(content="test", type="semantic")
            # Simulate the code path where metadata is None via getattr fallback
            item.__dict__["metadata"] = None
            return dspy.Prediction(memories=[item])

        extractor.extract = mock_extract  # ty:ignore[invalid-assignment]
        result = extractor.forward([{"role": "user", "content": "test"}])

        memories = result.memories
        assert len(memories) == 1
        assert memories[0][2] == {}
