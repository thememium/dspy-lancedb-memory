"""Tests for dspy_lancedb_memory.extraction — covering forward methods."""

from __future__ import annotations

import dspy

from dspy_lancedb_memory.extraction import MemoryReconciler
from dspy_lancedb_memory.models import ReconciledMemory

# ---------------------------------------------------------------------------
# MemoryReconciler.forward()
# ---------------------------------------------------------------------------


class TestMemoryReconciler:
    def test_forward_keep_action(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keep",
                    memory_id="existing-id",
                    final_content="existing content",
                    final_type="semantic",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="semantic",
            existing_memories=[
                {"id": "existing-id", "content": "existing content", "type": "semantic"}
            ],
        )

        assert result.reconciled.action == "keep"
        assert result.reconciled.memory_id == "existing-id"

    def test_forward_update_action(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="update",
                    memory_id="existing-id",
                    final_content="updated content",
                    final_type="semantic",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="semantic",
            existing_memories=[
                {"id": "existing-id", "content": "existing content", "type": "semantic"}
            ],
        )

        assert result.reconciled.action == "update"
        assert result.reconciled.final_content == "updated content"

    def test_forward_create_action(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="create",
                    memory_id="",
                    final_content="new content",
                    final_type="semantic",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="new content",
            new_memory_type="semantic",
            existing_memories=[],
        )

        assert result.reconciled.action == "create"
        assert result.reconciled.final_content == "new content"

    def test_forward_normalizes_action_prefixes(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keeping this memory",
                    memory_id="existing-id",
                    final_content="content",
                    final_type="semantic",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="semantic",
            existing_memories=[
                {"id": "existing-id", "content": "content", "type": "semantic"}
            ],
        )

        assert result.reconciled.action == "keep"

    def test_forward_unknown_action_defaults_to_create(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="something_unknown",
                    memory_id="",
                    final_content="content",
                    final_type="semantic",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="semantic",
            existing_memories=[],
        )

        assert result.reconciled.action == "create"

    def test_forward_keep_update_without_memory_id_uses_first_existing(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keep",
                    memory_id="",
                    final_content="content",
                    final_type="semantic",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="semantic",
            existing_memories=[
                {"id": "first-id", "content": "content", "type": "semantic"},
                {"id": "second-id", "content": "other", "type": "semantic"},
            ],
        )

        assert result.reconciled.memory_id == "first-id"

    def test_forward_create_without_final_content_uses_new_content(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="create",
                    memory_id="",
                    final_content="",
                    final_type="",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="new memory content",
            new_memory_type="semantic",
            existing_memories=[],
        )

        assert result.reconciled.final_content == "new memory content"
        assert result.reconciled.final_type == "semantic"

    def test_forward_keep_without_final_content_uses_existing(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keep",
                    memory_id="existing-id",
                    final_content="",
                    final_type="",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="semantic",
            existing_memories=[
                {
                    "id": "existing-id",
                    "content": "existing content",
                    "type": "preference",
                }
            ],
        )

        assert result.reconciled.final_content == "existing content"
        assert result.reconciled.final_type == "preference"

    def test_forward_keep_without_existing_or_content_uses_new_content(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keep",
                    memory_id="",
                    final_content="",
                    final_type="",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="new memory content",
            new_memory_type="semantic",
            existing_memories=[],
        )

        assert result.reconciled.action == "keep"
        assert result.reconciled.final_content == "new memory content"
        assert result.reconciled.final_type == "semantic"

    def test_forward_without_final_type_uses_new_type_for_create(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="create",
                    memory_id="",
                    final_content="content",
                    final_type="",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="episodic",
            existing_memories=[],
        )

        assert result.reconciled.final_type == "episodic"

    def test_forward_without_final_type_uses_existing_type_for_keep(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keep",
                    memory_id="existing-id",
                    final_content="content",
                    final_type="",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="episodic",
            existing_memories=[
                {"id": "existing-id", "content": "content", "type": "preference"}
            ],
        )

        assert result.reconciled.final_type == "preference"

    def test_forward_without_final_type_and_empty_existing_uses_new_type(self):
        reconciler = MemoryReconciler()

        def mock_reconcile(**kwargs):
            return dspy.Prediction(
                reconciled=ReconciledMemory(
                    action="keep",
                    memory_id="",
                    final_content="content",
                    final_type="",
                )
            )

        reconciler.reconcile = mock_reconcile  # ty:ignore[invalid-assignment]
        result = reconciler.forward(
            new_memory_content="test",
            new_memory_type="episodic",
            existing_memories=[],
        )

        assert result.reconciled.final_type == "episodic"
