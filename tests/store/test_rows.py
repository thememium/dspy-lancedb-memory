"""Tests for store.rows conversion and normalization helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_lancedb_memory.store.rows import normalize_extracted_memories, to_memory


class TestToMemory:
    def test_to_memory_with_relevance_score(self):
        row = {
            "id": "test-id",
            "user_id": "user-1",
            "session_id": "",
            "conversation_id": "",
            "memory_type": "semantic",
            "content": "test",
            "metadata": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "_relevance_score": 0.95,
        }
        memory = to_memory(row)
        assert memory.relevance_score == 0.95

    def test_to_memory_with_distance(self):
        row = {
            "id": "test-id",
            "user_id": "user-1",
            "session_id": "",
            "conversation_id": "",
            "memory_type": "semantic",
            "content": "test",
            "metadata": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "_distance": 0.3,
        }
        memory = to_memory(row)
        assert memory.relevance_score == pytest.approx(1.0 / 1.3)

    def test_to_memory_without_scores(self):
        row = {
            "id": "test-id",
            "user_id": "user-1",
            "session_id": "",
            "conversation_id": "",
            "memory_type": "semantic",
            "content": "test",
            "metadata": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }
        memory = to_memory(row)
        assert memory.relevance_score is None


def test_normalize_extracted_memories_with_tuples():
    """Test _normalize_extracted_memories with tuple inputs."""
    extracted = [
        ("content1", "semantic", {"source": "chat"}),
        ("content2", "preference"),
    ]
    result = normalize_extracted_memories(extracted)
    assert len(result) == 2
    assert result[0][0] == "content1"
    assert result[1][0] == "content2"


def test_normalize_extracted_memories_with_single_tuple():
    """Test _normalize_extracted_memories with 2-element tuple."""
    extracted = [("content1", "semantic")]
    result = normalize_extracted_memories(extracted)
    assert len(result) == 1


def test_normalize_extracted_memories_with_invalid_tuple():
    """Test _normalize_extracted_memories with 1-element tuple (skipped)."""
    extracted = [("content1",)]
    result = normalize_extracted_memories(extracted)
    assert len(result) == 0


def test_normalize_extracted_memories_deduplicates():
    """Test _normalize_extracted_memories deduplicates content."""
    extracted = [
        ("same content", "semantic"),
        ("same content", "preference"),
    ]
    result = normalize_extracted_memories(extracted)
    assert len(result) == 1


def test_normalize_extracted_memories_filters_empty():
    """Test _normalize_extracted_memories filters empty content."""
    extracted = [
        ("", "semantic"),
        ("  ", "semantic"),
        ("valid", "semantic"),
    ]
    result = normalize_extracted_memories(extracted)
    assert len(result) == 1
    assert result[0][0] == "valid"


def test_normalize_extracted_memories_with_objects():
    """Test _normalize_extracted_memories with object inputs."""
    obj = SimpleNamespace(
        content="test content",
        type="semantic",
        metadata={"source": "chat"},
    )
    result = normalize_extracted_memories([obj])
    assert len(result) == 1
    assert result[0][0] == "test content"
