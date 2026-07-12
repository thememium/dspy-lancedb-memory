"""Tests for dspy_lancedb_memory.models — covering remaining uncovered paths."""

from __future__ import annotations

import pytest

from dspy_lancedb_memory.models import (
    Memory,
    MemoryItem,
    MemoryOperation,
    MemoryType,
    PendingReconciliation,
    ReconciledMemory,
    Scope,
    memory_type_from_string,
)

# ---------------------------------------------------------------------------
# MemoryType._missing_() — lines 34-37
# ---------------------------------------------------------------------------


class TestMemoryTypeMissing:
    """Test MemoryType._missing_() which allows lookup by value string."""

    def test_memory_type_from_value_string_semantic(self):
        result = MemoryType("semantic")
        assert result == MemoryType.SEMANTIC
        assert result.value == "semantic"

    def test_memory_type_from_value_string_preference(self):
        result = MemoryType("preference")
        assert result == MemoryType.PREFERENCE

    def test_memory_type_from_value_string_episodic(self):
        result = MemoryType("episodic")
        assert result == MemoryType.EPISODIC

    def test_memory_type_from_value_string_procedural(self):
        result = MemoryType("procedural")
        assert result == MemoryType.PROCEDURAL

    def test_memory_type_from_value_string_summary(self):
        result = MemoryType("summary")
        assert result == MemoryType.SUMMARY

    def test_memory_type_from_value_string_artifact(self):
        result = MemoryType("artifact")
        assert result == MemoryType.ARTIFACT

    def test_memory_type_from_invalid_value_raises(self):
        with pytest.raises(ValueError):
            MemoryType("nonexistent_type")

    def test_memory_type_missing_returns_none_for_unknown(self):
        """Test _missing_ returns None for unknown values (line 36)."""
        # Call _missing_ directly to cover line 36
        result = MemoryType._missing_("nonexistent_type")
        assert result is None


# ---------------------------------------------------------------------------
# memory_type_from_string()
# ---------------------------------------------------------------------------


class TestMemoryTypeFromString:
    def test_none_returns_semantic_default(self):
        result = memory_type_from_string(None)
        assert result == MemoryType.SEMANTIC

    def test_enum_member_returns_same(self):
        result = memory_type_from_string(MemoryType.PREFERENCE)
        assert result == MemoryType.PREFERENCE

    def test_valid_string_returns_enum(self):
        result = memory_type_from_string("episodic")
        assert result == MemoryType.EPISODIC

    def test_custom_string_passes_through(self):
        result = memory_type_from_string("custom_type")
        assert result == "custom_type"

    def test_case_insensitive_lookup(self):
        result = memory_type_from_string("SEMANTIC")
        assert result == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# Scope model
# ---------------------------------------------------------------------------


class TestScope:
    def test_scope_to_dict(self):
        scope = Scope(tenant="acme", region="us-east")
        result = scope.to_dict()
        assert result == {"tenant": "acme", "region": "us-east"}

    def test_scope_with_extra_fields(self):
        scope = Scope(custom_field="value")
        assert getattr(scope, "custom_field") == "value"
        assert scope.to_dict() == {"custom_field": "value"}


# ---------------------------------------------------------------------------
# MemoryItem model
# ---------------------------------------------------------------------------


class TestMemoryItem:
    def test_memory_item_defaults(self):
        item = MemoryItem(content="test", type="semantic")
        assert item.metadata == {}

    def test_memory_item_with_metadata(self):
        item = MemoryItem(content="test", type="semantic", metadata={"source": "chat"})
        assert item.metadata == {"source": "chat"}


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------


class TestMemoryModel:
    def test_memory_relevance_score_default(self):
        memory = Memory(
            id="test-id",
            user_id="user-1",
            session_id="",
            conversation_id="",
            memory_type="semantic",
            content="test content",
            metadata={},
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert memory.relevance_score is None
        assert memory.is_active is True
        assert memory.replaces_id is None
        assert memory.scope == {}


# ---------------------------------------------------------------------------
# MemoryOperation model
# ---------------------------------------------------------------------------


class TestMemoryOperation:
    def test_memory_operation_defaults(self):
        op = MemoryOperation(action="create")
        assert op.content == ""
        assert op.search_query == ""
        assert op.memory_type == ""


# ---------------------------------------------------------------------------
# ReconciledMemory model
# ---------------------------------------------------------------------------


class TestReconciledMemory:
    def test_reconciled_memory_defaults(self):
        rm = ReconciledMemory(action="create")
        assert rm.memory_id == ""
        assert rm.final_content == ""
        assert rm.final_type == ""


# ---------------------------------------------------------------------------
# PendingReconciliation dataclass
# ---------------------------------------------------------------------------


class TestPendingReconciliation:
    def test_pending_reconciliation_defaults(self):
        rm = ReconciledMemory(action="create")
        pr = PendingReconciliation(
            content="test",
            inferred_type="semantic",
            decision=rm,
            user_id="user-1",
            session_id="",
            conversation_id="",
        )
        assert pr.scope == {}
        assert pr.metadata == {}
        assert pr.existing_row is None
