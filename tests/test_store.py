from __future__ import annotations

from types import SimpleNamespace

import pytest

from dspy_memory.store import LanceDSPyMemoryStore

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "their favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite food is pepperoni pizza": [0.95, 0.05, 0.0],
    "favorite color is blue": [0.99, 0.01, 0.0],
    "favorite color is red": [0.99, 0.01, 0.0],
    "favorite programming language is python": [0.0, 1.0, 0.0],
    "name is Edward": [0.9, 0.1, 0.0],
    "name is Edward Boswell": [0.88, 0.12, 0.0],
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


def _row_by_id(store: LanceDSPyMemoryStore, memory_id: str) -> dict:
    return next(row for row in _rows(store) if row["id"] == memory_id)


def test_upsert_semantic_skips_effective_duplicate_without_rewriting_content(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="their favorite food is pizza",
        memory_type="semantic",
        use_reconciler=False,
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    assert _row_by_id(store, original.id)["content"] == "favorite food is pizza"


def test_upsert_semantic_updates_refinement_in_place(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="semantic",
        similarity_threshold=0.85,
        use_reconciler=False,
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    updated = _row_by_id(store, original.id)
    assert updated["content"] == "favorite food is pepperoni pizza"
    assert updated["updated_at"] != updated["created_at"]


def test_upsert_semantic_does_not_overwrite_nonsemantic_match(
    store: LanceDSPyMemoryStore,
) -> None:
    preference = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="semantic",
        similarity_threshold=0.85,
        use_reconciler=False,
    )

    assert result.id != preference.id
    assert len(_rows(store)) == 2
    assert _row_by_id(store, preference.id)["content"] == "favorite food is pizza"
    assert _row_by_id(store, result.id)["memory_type"] == "semantic"


def test_upsert_semantic_uses_more_than_nearest_vector_candidate(
    store: LanceDSPyMemoryStore,
) -> None:
    distracting = store.create_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
    )
    target = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="semantic",
        similarity_threshold=0.5,
        use_reconciler=False,
    )

    assert result.id == target.id
    assert len(_rows(store)) == 2
    assert _row_by_id(store, target.id)["content"] == "favorite food is pepperoni pizza"
    assert _row_by_id(store, distracting.id)["content"] == "favorite color is blue"


def test_upsert_semantic_updates_conflicting_same_slot_value(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite color is red",
        memory_type="semantic",
        similarity_threshold=0.85,
        use_reconciler=False,
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    assert _row_by_id(store, original.id)["content"] == "favorite color is red"


def test_upsert_memories_extract_path_reuses_semantic_upsert_decision(
    store: LanceDSPyMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_memory.store.MemoryExtractor.forward",
        lambda self, messages: [("favorite food is pepperoni pizza", "semantic")],
    )

    result = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "favorite food is pepperoni pizza"}],
        extract=True,
        use_reconciler=False,
    )

    assert len(result) == 1
    assert len(_rows(store)) == 1
    assert _rows(store)[0]["content"] == "favorite food is pepperoni pizza"


@pytest.mark.parametrize("method_name", ["create_memories", "upsert_memories"])
def test_extract_paths_call_memory_extractor_module_instead_of_forward(
    store: LanceDSPyMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    class DummyExtractor:
        def __init__(self, signature=None):
            self.signature = signature

        def __call__(self, *, messages):
            return [("favorite programming language is python", "semantic")]

        def forward(self, messages):
            raise AssertionError("forward() should not be called directly")

    monkeypatch.setattr("dspy_memory.store.MemoryExtractor", DummyExtractor)

    method = getattr(store, method_name)
    result = method(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like Python."}],
        extract=True,
    )

    assert len(result) == 1
    assert result[0].content == "favorite programming language is python"


# ---------------------------------------------------------------------------
# MemoryReconciler integration tests
# ---------------------------------------------------------------------------


class StubReconciler:
    """Deterministic reconciler for unit tests.

    Returns *keep* when the new content exactly matches an existing one,
    *update* when the new content is a strict superset of an existing one,
    and *create* otherwise.
    """

    def __init__(self):
        pass

    def __call__(self, *, new_memory_content, new_memory_type, existing_memories):
        from dspy_memory.models import ReconciledMemory

        for existing in existing_memories:
            if existing["content"] == new_memory_content:
                return ReconciledMemory(
                    action="keep",
                    memory_id=existing["id"],
                    final_content=existing["content"],
                    final_type=existing["type"],
                )
            if new_memory_content.startswith(existing["content"] + " "):
                return ReconciledMemory(
                    action="update",
                    memory_id=existing["id"],
                    final_content=new_memory_content,
                    final_type=existing["type"],
                )
        return ReconciledMemory(
            action="create",
            memory_id="",
            final_content=new_memory_content,
            final_type=new_memory_type,
        )


def test_reconciler_keeps_exact_match(store, monkeypatch):
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr("dspy_memory.store.MemoryReconciler", StubReconciler)

    result = store.upsert_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    assert _row_by_id(store, original.id)["content"] == "name is Edward"


def test_reconciler_updates_refinement(store, monkeypatch):
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr("dspy_memory.store.MemoryReconciler", StubReconciler)

    result = store.upsert_memory(
        user_id="user-1",
        content="name is Edward Boswell",
        memory_type="semantic",
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    assert _row_by_id(store, original.id)["content"] == "name is Edward Boswell"


def test_reconciler_creates_unrelated_memory(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr("dspy_memory.store.MemoryReconciler", StubReconciler)

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
    )

    assert len(_rows(store)) == 2
    assert result.content == "favorite color is blue"


def test_reconciler_extract_path_consolidates_name(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_memory.store.MemoryExtractor.forward",
        lambda self, messages: [("name is Edward Boswell", "semantic")],
    )
    monkeypatch.setattr("dspy_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell."}],
        extract=True,
    )

    assert len(results) == 1
    assert len(_rows(store)) == 1
    assert _rows(store)[0]["content"] == "name is Edward Boswell"
