"""Embedded real-time engine (RTE) integration tests."""

from __future__ import annotations

import gc
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import lancedb
import pyarrow as pa

from dspy_lancedb_memory.store import LanceDSPyMemoryStore

DIM = 3

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite color is blue": [0.0, 1.0, 0.0],
    "I love hiking": [0.0, 0.0, 1.0],
    "hiking is my hobby": [0.0, 0.1, 0.9],
}


class StubStore(LanceDSPyMemoryStore):
    EMBEDDINGS = EMBEDDINGS

    def __init__(self, *, uri: str, table_name: str = "memories", **kwargs):
        super().__init__(
            uri=uri,
            table_name=table_name,
            embedding_lm=SimpleNamespace(model="test-embedding-model"),
            embedding_dim=DIM,
            reranker=None,
            **kwargs,
        )

    def _embed(self, text: str) -> list[float]:
        return self.EMBEDDINGS.get(text, [0.5, 0.5, 0.0])


def _legacy_table(uri: str, table_name: str) -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("conversation_id", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("content", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("replaces_id", pa.utf8(), nullable=True),
            pa.field("is_active", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
        ]
    )
    now = datetime.now(UTC).isoformat()
    db = lancedb.connect(uri)
    db.create_table(
        table_name,
        data=[
            {
                "id": "__seed__",
                "user_id": "__seed__",
                "session_id": "__seed__",
                "conversation_id": "__seed__",
                "memory_type": "seed",
                "content": "seed",
                "metadata": "{}",
                "created_at": now,
                "updated_at": now,
                "replaces_id": None,
                "is_active": True,
                "vector": [0.0] * DIM,
            }
        ],
        schema=schema,
    )


def test_store_enables_embedded_rte_with_indexes(tmp_path):
    store = StubStore(uri=str(tmp_path), table_name="memories")

    spec = store.table.get_rte_spec()
    assert spec is not None
    assert spec.spec_type == "unsharded"
    assert spec.writer_config_defaults == {"durable_write": "false"}
    assert store.table.schema.field("id").nullable is False

    indexes = {idx.name: idx.columns for idx in store.table.list_indices()}
    assert indexes["id_idx"] == ["id"]
    assert indexes["content_idx"] == ["content"]
    assert indexes["vector_idx"] == ["vector"]


def test_writes_are_immediately_visible(tmp_path):
    store = StubStore(uri=str(tmp_path))

    memory = store.create_memory(user_id="u1", content="favorite food is pizza")
    assert [m.id for m in store.list_memories(user_id="u1")] == [memory.id]

    store.update_memory(memory_id=memory.id, content="favorite color is blue")
    active = store.list_memories(user_id="u1")
    assert [m.content for m in active] == ["favorite color is blue"]

    store.delete_memory(memory_id=active[0].id)
    assert store.list_memories(user_id="u1") == []


def test_hybrid_search_works_on_rte_table(tmp_path):
    store = StubStore(uri=str(tmp_path))
    store.create_memory(user_id="u1", content="favorite food is pizza")
    store.create_memory(user_id="u1", content="I love hiking")

    results = store.search_memories(
        user_id="u1", query="hiking", query_type="hybrid", limit=5
    )

    assert results
    assert any("hiking" in memory.content for memory in results)


def test_durable_write_can_be_requested(tmp_path):
    store = StubStore(uri=str(tmp_path), rte_durable_write=True)

    assert store.table.get_rte_spec().writer_config_defaults == {
        "durable_write": "true"
    }


def test_flush_persists_writes_for_reopen(tmp_path):
    store = StubStore(uri=str(tmp_path))
    memory = store.create_memory(user_id="u1", content="I love hiking")
    store.flush()

    reopened = StubStore(uri=str(tmp_path))
    loaded = reopened.get_memory(memory_id=memory.id)
    assert loaded is not None
    assert loaded.content == "I love hiking"


def test_legacy_seed_table_is_migrated_to_rte(tmp_path):
    uri = str(tmp_path)
    _legacy_table(uri, "memories")

    store = StubStore(uri=uri)

    assert store.table.get_rte_spec() is not None
    assert store.table.schema.field("id").nullable is False
    assert store.table.search().where("id = '__seed__'").to_list() == []

    memory = store.create_memory(user_id="u1", content="I love hiking")
    loaded = store.get_memory(memory_id=memory.id)
    assert loaded is not None
    assert loaded.content == "I love hiking"


def test_atexit_flushes_writes(tmp_path):
    with patch("dspy_lancedb_memory.store.atexit.register") as mock_register:
        store = StubStore(uri=str(tmp_path))
        store.create_memory(user_id="u1", content="I love hiking")

    callback = mock_register.call_args[0][0]
    with patch.object(store, "flush") as mock_flush:
        callback()

    mock_flush.assert_called_once()


def test_atexit_flush_swallows_errors(tmp_path):
    with patch("dspy_lancedb_memory.store.atexit.register") as mock_register:
        store = StubStore(uri=str(tmp_path))

    callback = mock_register.call_args[0][0]
    with patch.object(store, "flush", side_effect=RuntimeError("boom")):
        callback()  # must not raise


def test_atexit_flush_skips_collected_store(tmp_path):
    with patch("dspy_lancedb_memory.store.atexit.register") as mock_register:
        store = StubStore(uri=str(tmp_path))

    callback = mock_register.call_args[0][0]
    del store
    gc.collect()

    callback()  # must not raise
