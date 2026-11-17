"""Low-level write primitives shared by memory mutations."""

from __future__ import annotations

import logging
from typing import Any

from .filters import escape_filter_value
from .rows import row_for_write

logger = logging.getLogger("dspy_lancedb_memory")


class WriteMixin:
    """Merge, soft-delete, flush, and raw-row reads for the memory store."""

    table: Any

    def _merge_rows(self, rows: list[dict[str, Any]]) -> None:
        """Upsert full rows through the RTE write path."""
        if not rows:
            return
        cleaned = [row_for_write(row, self.table.schema) for row in rows]
        self.table.merge_insert(
            "id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(cleaned)

    def _mark_inactive(self, rows: list[dict[str, Any]]) -> None:
        """Soft-delete rows, preserving every other column."""
        patched = [dict(row) for row in rows]
        for row in patched:
            row["is_active"] = False
        self._merge_rows(patched)

    def flush(self) -> None:
        """Flush buffered RTE writes to durable storage.

        Embedded RTE flushes its write-ahead log when the cached shard writers
        close.  Registered as an ``atexit`` hook, so normal process exit
        persists writes; call it explicitly before a planned shutdown.
        """
        self.table.close_rte_writers()

    def _get_raw_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch one row including its vector, active or not."""
        rows = (
            self.table.search()
            .where(f"id = '{escape_filter_value(memory_id)}'")
            .limit(1)
            .to_list()
        )
        return dict(rows[0]) if rows else None
