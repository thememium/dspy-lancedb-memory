"""Tests for dspy_lancedb_memory.extraction — covering forward methods."""

from __future__ import annotations

import dspy

from dspy_lancedb_memory.extraction import (
    MemoryExtractor,
    MemoryOperationExtractor,
    MemoryReconciler,
)
from dspy_lancedb_memory.models import MemoryItem, MemoryOperation, ReconciledMemory

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
