"""Conversation-driven create/update/delete operation processing."""

from __future__ import annotations

from typing import Any

import dspy

from ..extraction import MemoryOperationExtractor
from ..models import (
    Memories,
    Memory,
    MemoryOperation,
    MemoryType,
    ScopeLike,
    memory_type_from_string,
    memory_type_value,
)
from .filters import build_filters, filter_rows_by_json_fields
from .rows import to_memory


class ProcessMixin:
    """Apply extracted memory operations to the store."""

    table: Any
    _extraction_lm: dspy.LM | None

    _mark_inactive: Any
    delete_memories_by_search: Any
    upsert_memory: Any

    def process_memories(
        self,
        *,
        user_id: str,
        contents: list[dict[str, str]] | None = None,
        session_id: str = "",
        conversation_id: str = "",
        extract: bool = True,
        metadata: dict[str, Any] | None = None,
        scope: ScopeLike | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        skip_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> tuple[Memories, Memories]:
        """Extract and process memory operations (create/update/delete) from conversation.

        Returns a tuple of (created_or_updated_memories, deleted_memories).
        """
        if not contents:
            raise ValueError("contents is required")

        if extract:
            extractor = MemoryOperationExtractor()
            with dspy.context(lm=self._extraction_lm):
                operations = extractor(messages=contents).operations
        else:
            operations = [
                MemoryOperation(
                    action="create",
                    content=c["content"],
                )
                for c in contents
            ]

        created_or_updated: list[Memory] = []
        deleted: list[Memory] = []

        for op in operations:
            action = op.action.strip().lower()

            if action == "delete":
                search_query = op.search_query or op.content
                if not search_query:
                    continue

                if op.content:
                    exact_filters = build_filters(
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        memory_type=memory_type_value(op.memory_type),
                    )
                    exact_results = (
                        self.table.search()
                        .where(
                            " AND ".join(exact_filters)
                            + f" AND content = '{op.content}'"
                        )
                        .limit(10 if (scope or metadata_filter) else 1)
                        .to_list()
                    )
                    exact_results = filter_rows_by_json_fields(
                        [dict(row) for row in exact_results],
                        scope=scope,
                        metadata_filter=metadata_filter,
                    )
                    if exact_results:
                        self._mark_inactive(exact_results[:1])
                        deleted.extend(
                            to_memory(dict(row)) for row in exact_results[:1]
                        )
                        continue

                removed = self.delete_memories_by_search(
                    user_id=user_id,
                    query=search_query,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=memory_type_value(op.memory_type),
                    scope=scope,
                    metadata_filter=metadata_filter,
                    similarity_threshold=similarity_threshold,
                    limit=1,
                )
                deleted.extend(removed)
                continue

            if action in ("create", "update") and op.content:
                created_or_updated.append(
                    self.upsert_memory(
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        content=op.content,
                        memory_type=(
                            memory_type_from_string(op.memory_type)
                            if op.memory_type
                            else MemoryType.SEMANTIC
                        ),
                        metadata=metadata,
                        scope=scope,
                        similarity_threshold=similarity_threshold,
                        skip_threshold=skip_threshold,
                        use_reconciler=use_reconciler,
                    )
                )

        return created_or_updated, deleted
