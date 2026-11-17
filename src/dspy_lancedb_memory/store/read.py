"""Memory reads, history, metadata updates, creation, and deletion."""

from __future__ import annotations

import logging
from typing import Any, cast

import dspy

from ..extraction import MemoryExtractor
from ..models import (
    Memories,
    Memory,
    MemoryType,
    ScopeLike,
    memory_type_from_string,
    memory_type_value,
)
from .bound import BoundMemoryStore
from .filters import build_filters, escape_filter_value, filter_rows_by_json_fields
from .rows import (
    build_memory_row,
    carry_legacy_fields,
    normalize_extracted_memories,
    to_memory,
    without_vectors,
)
from .search import filter_by_relevance
from .serialization import merge_metadata, scope_dict, split_metadata_scope

logger = logging.getLogger("dspy_lancedb_memory")


class ReadMixin:
    """CRUD operations for the memory store."""

    table: Any
    _extraction_lm: dspy.LM | None
    _extraction_signature: type[dspy.Signature]

    _embed: Any
    _get_raw_memory: Any
    _merge_rows: Any
    _mark_inactive: Any
    _select_all: Any

    def _build_memory_row(
        self,
        *,
        user_id: str,
        content: str,
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str,
        metadata: dict[str, Any] | None,
        scope: ScopeLike | None = None,
    ) -> dict[str, Any]:
        return build_memory_row(
            embed=self._embed,
            user_id=user_id,
            content=content,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=memory_type,
            metadata=metadata,
            scope=scope,
        )

    def _get_raw_memory(self, memory_id: str) -> dict[str, Any] | None: ...

    def _merge_rows(self, rows: list[dict[str, Any]]) -> None: ...

    def _mark_inactive(self, rows: list[dict[str, Any]]) -> None: ...

    def with_scope(
        self,
        *,
        user_id: str,
        session_id: str = "",
        conversation_id: str = "",
        scope: ScopeLike | None = None,
    ) -> BoundMemoryStore:
        return BoundMemoryStore(
            cast(Any, self),
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            scope=scope_dict(scope),
        )

    def get_memory(
        self, *, memory_id: str, include_inactive: bool = False
    ) -> Memory | None:
        filters = [f"id = '{escape_filter_value(memory_id)}'"]
        if not include_inactive:
            filters.append("is_active = true")
        rows = self.table.search().where(" AND ".join(filters)).limit(1).to_list()
        if not rows:
            return None
        return to_memory(dict(rows[0]))

    def list_memories(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        conversation_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        scope: ScopeLike | None = None,
        metadata_filter: dict[str, Any] | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> Memories:
        filters = build_filters(
            user_id=user_id,
            session_id=session_id or "",
            conversation_id=conversation_id or "",
            memory_type=memory_type_value(memory_type),
            active_only=not include_inactive,
        )
        query = self.table.search().where(" AND ".join(filters))
        if limit is not None:
            query = query.limit(limit * (10 if (scope or metadata_filter) else 1))
        rows = [dict(row) for row in query.to_list()]
        rows = filter_rows_by_json_fields(
            rows,
            scope=scope_dict(scope),
            metadata_filter=metadata_filter,
        )
        if limit is not None:
            rows = rows[:limit]
        return without_vectors(rows)

    def get_memory_history(self, *, memory_id: str) -> Memories:
        rows = [dict(row) for row in self.table.search().to_list()]
        by_id = {str(row["id"]): row for row in rows}
        if memory_id not in by_id:
            return []

        chain_ids: list[str] = []
        current_id: str | None = memory_id
        while current_id and current_id in by_id and current_id not in chain_ids:
            chain_ids.append(current_id)
            replaces_id = by_id[current_id].get("replaces_id")
            current_id = str(replaces_id) if replaces_id else None

        changed = True
        while changed:
            changed = False
            known = set(chain_ids)
            for row_id, row in by_id.items():
                if row_id not in known and row.get("replaces_id") in known:
                    chain_ids.insert(0, row_id)
                    changed = True

        return without_vectors([by_id[row_id] for row_id in chain_ids])

    def update_memory_metadata(
        self,
        *,
        memory_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> Memory | None:
        old_row = self._get_raw_memory(memory_id)
        if old_row is None or not old_row.get("is_active", True):
            return None
        old = to_memory(old_row)
        next_metadata = dict(old.metadata) if merge else {}
        next_metadata.update(metadata)
        new_row = self._build_memory_row(
            user_id=old.user_id,
            content=old.content,
            session_id=old.session_id,
            conversation_id=old.conversation_id,
            memory_type=old.memory_type,
            metadata=next_metadata,
            scope=old.scope,
        )
        new_row["replaces_id"] = old.id
        carry_legacy_fields(new_row, old_row, keep_search_text=True)
        old_row["is_active"] = False
        self._merge_rows([new_row, old_row])
        return to_memory(new_row)

    def update_memory_scope(
        self,
        *,
        memory_id: str,
        scope: ScopeLike,
        merge: bool = True,
    ) -> Memory | None:
        old_row = self._get_raw_memory(memory_id)
        if old_row is None or not old_row.get("is_active", True):
            return None
        old = to_memory(old_row)
        next_scope = dict(old.scope) if merge else {}
        next_scope.update(scope_dict(scope))
        new_row = self._build_memory_row(
            user_id=old.user_id,
            content=old.content,
            session_id=old.session_id,
            conversation_id=old.conversation_id,
            memory_type=old.memory_type,
            metadata=old.metadata,
            scope=next_scope,
        )
        new_row["replaces_id"] = old.id
        carry_legacy_fields(new_row, old_row, keep_search_text=True)
        old_row["is_active"] = False
        self._merge_rows([new_row, old_row])
        return to_memory(new_row)

    def create_memories(
        self,
        *,
        user_id: str,
        contents: list[dict[str, str]] | None = None,
        session_id: str = "",
        conversation_id: str = "",
        extract: bool = True,
        memory_type: MemoryType | str | None = None,
        metadata: dict[str, Any] | None = None,
        scope: ScopeLike | None = None,
    ) -> Memories:
        """
        Store content for a user — either raw or extracted from conversation.

        Parameters
        ----------
        user_id : str
            Arbitrary identifier for the user this memory belongs to.
        contents : list[dict[str, str]] | None
            A conversation turn in standard ``{"role": ..., "content": ...}`` format.
            Required when *extract* is ``True`` (default).
        session_id : str
            Optional session / thread identifier for grouping.
        conversation_id : str
            Optional grouping key (e.g. conversation / dialogue ID).
        extract : bool
            When ``True``, a DSPy Signature is used to extract **all** salient
            memories from *contents*. When ``False``, *contents* must contain
            exactly one item and is stored verbatim.
        memory_type : MemoryType | str | None
            Force a specific memory category. If ``None`` while *extract* is ``True``,
            the LLM chooses categories. If ``None`` while *extract* is ``False``,
            falls back to ``MemoryType.SEMANTIC``.
        metadata : dict[str, Any] | None
            Structured data attached to every new memory row but not embedded.
        scope : dict[str, Any] | None
            Custom ownership/query dimensions attached to every new memory row.

        Returns
        -------
        Memories
            The full rows written to LanceDB.
        """
        if extract:
            if not contents:
                raise ValueError("contents is required when extract=True")

            extractor = MemoryExtractor(signature=self._extraction_signature)
            with dspy.context(lm=self._extraction_lm):
                extracted = normalize_extracted_memories(
                    cast(list[Any], extractor(messages=contents).memories)
                )

            stored = [
                self.create_memory(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    content=content,
                    memory_type=(
                        memory_type_from_string(memory_type)
                        if memory_type is not None
                        else inferred_type
                    ),
                    metadata=merge_metadata(metadata, item_metadata),
                    scope=scope,
                )
                for content, inferred_type, item_metadata in extracted
            ]
            return without_vectors(stored)

        if not contents or len(contents) != 1:
            raise ValueError(
                "verbatim storage (extract=False) requires exactly one item in contents"
            )

        row = self.create_memory(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            content=contents[0]["content"],
            memory_type=(
                memory_type_from_string(memory_type)
                if memory_type is not None
                else MemoryType.SEMANTIC
            ),
            metadata=metadata,
            scope=scope,
        )
        return without_vectors([row])

    def create_memory(
        self,
        *,
        user_id: str,
        content: str,
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
        scope: ScopeLike | None = None,
    ) -> Memory:
        row = self._build_memory_row(
            user_id=user_id,
            content=content,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=memory_type,
            metadata=metadata,
            scope=scope,
        )
        self._merge_rows([row])
        return to_memory(row)

    def update_memory(self, *, memory_id: str, content: str) -> None:
        old_record = self._get_raw_memory(memory_id)
        if old_record is None:
            return
        if not old_record.get("is_active", True):
            logger.warning(
                "update_memory called on inactive record %s — skipping", memory_id
            )
            return

        metadata, scope = split_metadata_scope(old_record.get("metadata"))
        new_row = self._build_memory_row(
            user_id=old_record["user_id"],
            content=content,
            session_id=old_record.get("session_id", ""),
            conversation_id=old_record.get("conversation_id", ""),
            memory_type=old_record.get("memory_type", "semantic"),
            metadata=metadata,
            scope=scope,
        )
        new_row["replaces_id"] = memory_id
        carry_legacy_fields(new_row, old_record, keep_search_text=False)
        old_record["is_active"] = False

        self._merge_rows([new_row, old_record])

    def delete_memory(self, *, memory_id: str) -> None:
        row = self._get_raw_memory(memory_id)
        if row is None:
            return
        self._mark_inactive([row])

    def delete_memories_by_search(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str | None = None,
        scope: ScopeLike | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        limit: int = 5,
    ) -> Memories:
        filters = build_filters(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=memory_type_value(memory_type),
        )

        fetch_limit = limit * (10 if (scope or metadata_filter) else 1)
        candidates = (
            self._select_all(
                self.table.search(self._embed(query), vector_column_name="vector"),
                "_distance",
            )
            .where(" AND ".join(filters))
            .limit(fetch_limit)
            .to_list()
        )

        if not candidates:
            return []

        results = filter_rows_by_json_fields(
            [dict(row) for row in candidates],
            scope=scope,
            metadata_filter=metadata_filter,
        )
        results = filter_by_relevance(results, min_score=similarity_threshold)
        results = results[:limit]

        if not results:
            return []

        self._mark_inactive(results)
        return [to_memory(dict(row)) for row in results]
