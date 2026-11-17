"""LanceDB table creation, schema, indexes, and the write engine."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import pyarrow as pa
from lancedb import RTEWriteSpec
from lancedb.index import FTS, BTree, IvfPq

logger = logging.getLogger("dspy_lancedb_memory")

# Embedded RTE allows one writer per shard per dataset.  Multiple stores for
# the same (uri, table) must share one table handle or their writers fence
# each other with increasing epochs.
TABLE_CACHE: dict[tuple[str, str], Any] = {}
TABLE_CACHE_LOCK = threading.RLock()


def table_cache_key(uri: str, table_name: str) -> tuple[str, str]:
    if "://" in uri:
        return (uri, table_name)
    return (os.path.abspath(uri), table_name)


def get_vector_dim(table) -> int | None:
    """Return the fixed vector width of *table*, if declared."""
    for field in table.schema:
        if field.name == "vector":
            field_type = field.type
            if isinstance(field_type, pa.FixedSizeListType):
                return field_type.list_size
            if isinstance(field_type, pa.ListType):
                return None
    return None


def has_index(table, column: str) -> bool:
    return any(column in (index.columns or []) for index in table.list_indices())


def memory_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("user_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("conversation_id", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("content", pa.string()),
            pa.field("metadata", pa.string()),  # JSON-encoded dict
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("replaces_id", pa.utf8(), nullable=True),
            pa.field("is_active", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
        ]
    )


def create_fts_index(table) -> None:
    """Create a full-text search index on the ``content`` column if absent."""
    try:
        if not has_index(table, "content"):
            table.create_index("content", config=FTS())
    except Exception:
        logger.debug("FTS index already exists or could not be created", exc_info=True)


def ensure_rte(table, *, durable_write: bool) -> None:
    """Enable the embedded real-time engine for millisecond writes.

    RTE needs a non-nullable primary key and a scalar index on it; both are
    created here for tables made by older versions.  Tables that already
    have an RTE spec are left untouched.
    """
    # RTE refuses vector search unless the vector index is maintained by
    # the write spec; the default spec maintains every index on the table.
    if not has_index(table, "vector"):
        table.create_index("vector", config=IvfPq())
    if table.get_rte_spec() is not None:
        return
    if table.schema.field("id").nullable:
        table.alter_columns({"path": "id", "nullable": False})
    # Older versions needed a seed row to create an empty table; drop it
    # while the table is still on the regular write path.
    table.delete("id = '__seed__'")
    table.set_unenforced_primary_key("id")
    if not has_index(table, "id"):
        table.create_index("id", config=BTree())
    spec = RTEWriteSpec.unsharded().with_writer_config_defaults(
        {"durable_write": "true" if durable_write else "false"}
    )
    table.enable_rte(spec)


class TableMixin:
    """Table lifecycle and index maintenance for the memory store."""

    db: Any
    table_uri: str
    table_name: str
    embedding_dim: int | None
    rte_durable_write: bool

    def _table_cache_key(self) -> tuple[str, str]:
        return table_cache_key(self.table_uri, self.table_name)

    def _get_or_create_table(self):
        assert self.embedding_dim is not None, (
            "embedding_dim must be set before creating table"
        )
        key = self._table_cache_key()
        with TABLE_CACHE_LOCK:
            table = TABLE_CACHE.get(key)
            if table is not None:
                existing_dim = get_vector_dim(table)
                if existing_dim is not None and existing_dim != self.embedding_dim:
                    logger.warning(
                        "Table %r vector dimension (%d) does not match "
                        "embedding_dim (%d); dropping and recreating.",
                        self.table_name,
                        existing_dim,
                        self.embedding_dim,
                    )
                    self.db.drop_table(self.table_name)
                    TABLE_CACHE.pop(key, None)
                else:
                    self._ensure_fts_index(table)
                    self._ensure_rte(table)
                    return table

            if self.table_name in self.db.list_tables():
                table = self.db.open_table(self.table_name)
                existing_dim = get_vector_dim(table)
                if existing_dim is not None and existing_dim != self.embedding_dim:
                    logger.warning(
                        "Table %r vector dimension (%d) does not match "
                        "embedding_dim (%d); dropping and recreating.",
                        self.table_name,
                        existing_dim,
                        self.embedding_dim,
                    )
                    self.db.drop_table(self.table_name)
                else:
                    self._ensure_fts_index(table)
                    self._ensure_rte(table)
                    TABLE_CACHE[key] = table
                    return table

            table = self.db.create_table(
                self.table_name, schema=memory_schema(self.embedding_dim)
            )
            self._ensure_fts_index(table)
            self._ensure_rte(table)
            TABLE_CACHE[key] = table
            return table

    def _ensure_fts_index(self, table) -> None:
        create_fts_index(table)

    def _ensure_rte(self, table) -> None:
        ensure_rte(table, durable_write=self.rte_durable_write)
