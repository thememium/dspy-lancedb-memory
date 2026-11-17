from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest
from pydantic import BaseModel

from dspy_lancedb_memory.models import Scope
from dspy_lancedb_memory.store import LanceDSPyMemoryStore, rows

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "their favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite food is pepperoni pizza": [0.95, 0.05, 0.0],
    "favorite color is blue": [0.99, 0.01, 0.0],
    "favorite color is red": [0.99, 0.01, 0.0],
    "favorite programming language is python": [0.0, 1.0, 0.0],
    "name is Edward": [0.9, 0.1, 0.0],
    "name is Edward Boswell": [0.88, 0.12, 0.0],
    "I love hiking": [0.0, 0.0, 1.0],
    "hiking is my hobby": [0.0, 0.0, 0.95],
    "enjoys outdoor activities": [0.0, 0.1, 0.9],
    "what are hobbies": [0.0, 0.05, 0.95],
    "what food do I like": [0.95, 0.0, 0.05],
    "car color is blue": [0.0, 0.5, 0.5],
    "car color is red": [0.0, 0.5, 0.5],
    # skip-threshold test vectors — L2 distance ≈ 0.057 with "I love hiking" (1 - dist ≈ 0.94)
    "hiking is enjoyable": [0.05, 0.02, 0.98],
    # skip-threshold test vectors — L2 distance ≈ 0.028 with "car color is blue" (1 - dist ≈ 0.97)
    "vehicle color blue": [0.02, 0.48, 0.5],
    # skip-threshold test vector — L2 distance ≈ 0.094 with "car color is blue" (1 - dist ≈ 0.91, >= 0.85 skip)
    "my car is blue": [0.08, 0.45, 0.5],
    "hiking": [0.0, 0.05, 0.95],
    # For reconciler test — similar to "name is Edward" but not identical
    "my name is edward": [0.85, 0.15, 0.0],
}


class StubMemoryStore(LanceDSPyMemoryStore):
    def __init__(
        self, *, uri: str, table_name: str, embeddings: dict[str, list[float]]
    ):
        self._embeddings = embeddings
        super().__init__(
            uri=uri,
            table_name=table_name,
            embedding_lm=SimpleNamespace(model="test-embedding-model"),
            embedding_dim=3,
            reranker=None,
        )

    def _embed(self, text: str) -> list[float]:
        return self._embeddings[text]


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> LanceDSPyMemoryStore:
    return StubMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )


def _rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__'").to_list()


def _active_rows(store: LanceDSPyMemoryStore) -> list[dict]:
    return store.table.search().where("id != '__seed__' AND is_active = true").to_list()


def _row_by_id(store: LanceDSPyMemoryStore, memory_id: str) -> dict:
    return next(row for row in _rows(store) if row["id"] == memory_id)


# ---------------------------------------------------------------------------
# Custom scope / metadata filters
# ---------------------------------------------------------------------------


def test_search_memories_filters_by_scope(store: LanceDSPyMemoryStore) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "tenant-a", "repo": "alpha"},
    )
    scoped = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "tenant-a", "repo": "beta"},
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        scope={"repo": "beta"},
    )

    assert [r.id for r in results] == [scoped.id]
    assert results[0].scope == {"tenant": "tenant-a", "repo": "beta"}


def test_upsert_memory_respects_scope_boundaries(store: LanceDSPyMemoryStore) -> None:
    original = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "tenant-a"},
        use_reconciler=False,
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"project_id": "beta"},
        use_reconciler=False,
    )

    assert result.id != original.id
    assert len(_rows(store)) == 2
    assert result.scope == {"project_id": "beta"}


def test_search_memories_filters_by_metadata(store: LanceDSPyMemoryStore) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata={"source": "chat"},
    )
    doc_memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata={"source": "doc", "priority": "high"},
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        metadata_filter={"source": "doc"},
    )

    assert [r.id for r in results] == [doc_memory.id]
    assert results[0].metadata == {"source": "doc", "priority": "high"}


def test_update_memory_preserves_scope(store: LanceDSPyMemoryStore) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"project_id": "alpha"},
    )

    store.update_memory(
        memory_id=original.id,
        content="favorite food is pepperoni pizza",
    )

    active = _active_rows(store)
    assert len(active) == 1
    updated = rows.without_vectors([active[0]])[0]
    assert updated.scope == {"project_id": "alpha"}


def test_create_memories_merges_extracted_metadata(
    store: LanceDSPyMemoryStore, monkeypatch
) -> None:
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.read.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                (
                    "favorite programming language is python",
                    "semantic",
                    {"source": "extractor", "confidence": 0.9},
                )
            ]
        ),
    )

    results = store.create_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like Python."}],
        metadata={"batch": "b1"},
        scope={"project_id": "alpha"},
    )

    assert len(results) == 1
    assert results[0].metadata == {
        "batch": "b1",
        "source": "extractor",
        "confidence": 0.9,
    }
    assert results[0].scope == {"project_id": "alpha"}


# ---------------------------------------------------------------------------
# Developer-experience helpers
# ---------------------------------------------------------------------------


def test_scope_model_is_accepted_by_search(store: LanceDSPyMemoryStore) -> None:
    scoped = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope=Scope(tenant="tenant-a", repo="memory-sdk", environment="prod"),
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        scope=Scope(repo="memory-sdk", environment="prod"),
    )

    assert [r.id for r in results] == [scoped.id]
    assert results[0].scope == {
        "tenant": "tenant-a",
        "repo": "memory-sdk",
        "environment": "prod",
    }


def test_custom_pydantic_scope_model_is_accepted(
    store: LanceDSPyMemoryStore,
) -> None:
    class DeploymentScope(BaseModel):
        tenant: str
        repo: str
        environment: str

    scoped = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope=DeploymentScope(
            tenant="tenant-a",
            repo="memory-sdk",
            environment="prod",
        ),
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        scope=DeploymentScope(
            tenant="tenant-a",
            repo="memory-sdk",
            environment="prod",
        ),
    )

    assert [r.id for r in results] == [scoped.id]
    assert results[0].scope == {
        "tenant": "tenant-a",
        "repo": "memory-sdk",
        "environment": "prod",
    }


def test_bound_store_applies_context(store: LanceDSPyMemoryStore) -> None:
    bound = store.with_scope(
        user_id="user-1",
        session_id="session-1",
        scope={"tenant": "tenant-a", "repo": "memory-sdk"},
    )
    memory = bound.create_memory(
        content="favorite food is pizza",
        memory_type="semantic",
    )

    results = bound.search_memories(query="what food do I like")

    assert [r.id for r in results] == [memory.id]
    assert results[0].session_id == "session-1"
    assert results[0].scope == {"tenant": "tenant-a", "repo": "memory-sdk"}


def test_metadata_filter_operators(store: LanceDSPyMemoryStore) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata={"priority": "low", "confidence": 0.4, "tags": ["chat"]},
    )
    target = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata={
            "priority": "high",
            "confidence": 0.9,
            "tags": ["doc", "profile"],
        },
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        metadata_filter={
            "priority": {"in": ["high", "urgent"]},
            "confidence": {"gte": 0.8},
            "tags": {"contains": "profile"},
        },
    )

    assert [r.id for r in results] == [target.id]


def test_reserved_scope_metadata_key_is_rejected(store: LanceDSPyMemoryStore) -> None:
    with pytest.raises(ValueError, match="reserved"):
        store.create_memory(
            user_id="user-1",
            content="favorite food is pizza",
            memory_type="semantic",
            metadata={"_scope": {"project_id": "alpha"}},
        )


def test_filter_values_with_quotes_do_not_break_queries(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user'1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user'1",
        query="what food do I like",
    )

    assert [r.id for r in results] == [memory.id]


# ---------------------------------------------------------------------------
# Group 1: BoundMemoryStore methods beyond create/search
# ---------------------------------------------------------------------------


def test_bound_store_upsert_memory_applies_context(
    store: LanceDSPyMemoryStore,
) -> None:
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    original = bound.upsert_memory(
        content="favorite food is pizza",
        memory_type="semantic",
        use_reconciler=False,
    )

    same = bound.upsert_memory(
        content="favorite food is pizza",
        memory_type="semantic",
        use_reconciler=False,
    )

    assert same.id == original.id
    assert same.scope == {"tenant": "tenant-a"}

    different_scope = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-b"},
    )
    new = different_scope.upsert_memory(
        content="favorite food is pizza",
        memory_type="semantic",
        use_reconciler=False,
    )

    assert new.id != original.id
    assert len(_rows(store)) == 2
    assert new.scope == {"tenant": "tenant-b"}


def test_bound_store_create_memories_applies_context(
    store: LanceDSPyMemoryStore,
) -> None:
    bound = store.with_scope(
        user_id="user-1",
        session_id="session-1",
        scope={"project_id": "alpha"},
    )

    results = bound.create_memories(
        contents=[{"role": "user", "content": "favorite food is pizza"}],
        extract=False,
        memory_type="semantic",
    )

    assert len(results) == 1
    assert results[0].session_id == "session-1"
    assert results[0].scope == {"project_id": "alpha"}


def test_bound_store_list_memories_applies_context(
    store: LanceDSPyMemoryStore,
) -> None:
    bound_a = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )
    bound_b = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-b"},
    )

    mem_a = bound_a.create_memory(
        content="favorite food is pizza",
        memory_type="semantic",
    )
    bound_b.create_memory(
        content="favorite color is blue",
        memory_type="semantic",
    )

    listed = bound_a.list_memories()

    assert len(listed) == 1
    assert listed[0].id == mem_a.id
    assert listed[0].scope == {"tenant": "tenant-a"}


# ---------------------------------------------------------------------------
# Group 5: ScopeLike dispatch for upsert + BoundMemoryStore
# ---------------------------------------------------------------------------


def test_upsert_memory_with_pydantic_base_model_scope(
    store: LanceDSPyMemoryStore,
) -> None:
    class DeploymentScope(BaseModel):
        tenant: str
        region: str

    scope = DeploymentScope(tenant="acme", region="us-east")

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope=scope,
        use_reconciler=False,
    )

    assert result.scope == {"tenant": "acme", "region": "us-east"}

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        scope=DeploymentScope(tenant="acme", region="us-east"),
    )

    assert [r.id for r in results] == [result.id]


def test_bound_store_with_pydantic_base_model_scope(
    store: LanceDSPyMemoryStore,
) -> None:
    class ProjectScope(BaseModel):
        team: str
        sprint: str

    bound = store.with_scope(
        user_id="user-1",
        scope=ProjectScope(team="backend", sprint="s42"),
    )

    memory = bound.create_memory(
        content="favorite food is pizza",
        memory_type="semantic",
    )

    assert memory.scope == {"team": "backend", "sprint": "s42"}

    listed = bound.list_memories()
    assert len(listed) == 1
    assert listed[0].scope == {"team": "backend", "sprint": "s42"}
