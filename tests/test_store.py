from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest

from dspy_lancedb_memory.store import LanceDSPyMemoryStore

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

    assert result.id != original.id
    assert len(_active_rows(store)) == 1
    assert result.content == "favorite food is pepperoni pizza"
    old = _row_by_id(store, original.id)
    assert old["is_active"] is False


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

    assert result.id != target.id
    assert len(_active_rows(store)) == 2
    assert result.content == "favorite food is pepperoni pizza"
    assert _row_by_id(store, distracting.id)["content"] == "favorite color is blue"
    assert _row_by_id(store, target.id)["is_active"] is False


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

    assert result.id != original.id
    assert len(_active_rows(store)) == 1
    assert result.content == "favorite color is red"
    assert _row_by_id(store, original.id)["is_active"] is False


def test_upsert_memories_extract_path_reuses_semantic_upsert_decision(
    store: LanceDSPyMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite food is pepperoni pizza", "semantic")]
        ),
    )

    result = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "favorite food is pepperoni pizza"}],
        extract=True,
        use_reconciler=False,
    )

    assert len(result) == 1
    assert len(_active_rows(store)) == 1
    assert result[0].content == "favorite food is pepperoni pizza"


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
            return dspy.Prediction(
                memories=[("favorite programming language is python", "semantic")]
            )

        def forward(self, messages):
            raise AssertionError("forward() should not be called directly")

    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryExtractor", DummyExtractor)

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
        from dspy_lancedb_memory.models import ReconciledMemory

        for existing in existing_memories:
            if existing["content"] == new_memory_content:
                return dspy.Prediction(
                    reconciled=ReconciledMemory(
                        action="keep",
                        memory_id=existing["id"],
                        final_content=existing["content"],
                        final_type=existing["type"],
                    )
                )
            if new_memory_content.startswith(existing["content"] + " "):
                return dspy.Prediction(
                    reconciled=ReconciledMemory(
                        action="update",
                        memory_id=existing["id"],
                        final_content=new_memory_content,
                        final_type=existing["type"],
                    )
                )
        return dspy.Prediction(
            reconciled=ReconciledMemory(
                action="create",
                memory_id="",
                final_content=new_memory_content,
                final_type=new_memory_type,
            )
        )


def test_reconciler_keeps_exact_match(store, monkeypatch):
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

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

    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    result = store.upsert_memory(
        user_id="user-1",
        content="name is Edward Boswell",
        memory_type="semantic",
        skip_threshold=1.0,
    )

    assert result.id != original.id
    assert len(_active_rows(store)) == 1
    assert result.content == "name is Edward Boswell"
    assert _row_by_id(store, original.id)["is_active"] is False


def test_reconciler_creates_unrelated_memory(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite color is blue",
        memory_type="semantic",
        skip_threshold=1.0,
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
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell."}],
        extract=True,
        skip_threshold=1.0,
    )

    assert len(results) == 1
    assert len(_active_rows(store)) == 1
    assert results[0].content == "name is Edward Boswell"


# ---------------------------------------------------------------------------
# Parallel reconciliation tests
# ---------------------------------------------------------------------------


def test_parallel_reconciler_same_as_sequential(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results_parallel = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=4,
    )

    assert len(results_parallel) == 1
    assert len(_active_rows(store)) == 1
    assert results_parallel[0].content == "name is Edward Boswell"


def test_parallel_reconciler_respects_skip_threshold(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("hiking is enjoyable", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "hiking is enjoyable"}],
        extract=True,
        skip_threshold=0.9,
        num_threads=4,
    )

    assert len(results) == 1
    assert results[0].content == "I love hiking"


def test_parallel_num_threads_parameter(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("favorite food is pizza", "semantic"),
                ("favorite color is blue", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like pizza and blue."}],
        extract=True,
        num_threads=2,
    )

    assert len(results) == 2


def test_parallel_with_multiple_memories(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("favorite food is pizza", "semantic"),
                ("favorite color is blue", "semantic"),
                ("favorite programming language is python", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like pizza, blue, and python."}],
        extract=True,
        num_threads=4,
    )

    assert len(results) == 3
    contents = {r.content for r in results}
    assert "favorite food is pizza" in contents
    assert "favorite color is blue" in contents
    assert "favorite programming language is python" in contents


def test_parallel_keep_action(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My name is Edward."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=4,
    )

    assert len(results) == 1
    assert results[0].content == "name is Edward"
    assert len(_rows(store)) == 1


def test_parallel_create_action(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite color is blue", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My favorite color is blue."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=4,
    )

    assert len(results) == 1
    assert results[0].content == "favorite color is blue"
    assert len(_rows(store)) == 2


def test_parallel_update_action(store, monkeypatch):
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=4,
    )

    assert len(results) == 1
    assert results[0].content == "name is Edward Boswell"
    assert results[0].id != original.id
    assert len(_active_rows(store)) == 1
    assert _row_by_id(store, original.id)["is_active"] is False


def test_parallel_mixed_actions(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("name is Edward", "semantic"),
                ("favorite color is blue", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I'm Edward and I like blue."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=4,
    )

    assert len(results) == 2
    contents = {r.content for r in results}
    assert "name is Edward" in contents
    assert "favorite color is blue" in contents
    assert len(_rows(store)) == 2


def test_parallel_num_threads_one_is_sequential(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=1,
    )

    assert len(results) == 1
    assert results[0].content == "name is Edward Boswell"


def test_parallel_deduplicates_extracted_memories(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("favorite food is pizza", "semantic"),
                ("favorite food is pizza", "semantic"),
                ("favorite color is blue", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like pizza and blue."}],
        extract=True,
        num_threads=4,
    )

    assert len(results) == 2


def test_parallel_empty_extraction(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(memories=[]),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "nothing memorable"}],
        extract=True,
        num_threads=4,
    )

    assert len(results) == 0


def test_parallel_user_scoping(store, monkeypatch):
    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = store.upsert_memories(
        user_id="user-2",
        contents=[{"role": "user", "content": "My name is Edward."}],
        extract=True,
        skip_threshold=1.0,
        num_threads=4,
    )

    assert len(results) == 1
    assert results[0].content == "name is Edward"
    assert len(_rows(store)) == 2


# ---------------------------------------------------------------------------
# Regression tests — capture CURRENT behavior before refactoring
# ---------------------------------------------------------------------------


def test_regression_create_memory_increases_row_count(
    store: LanceDSPyMemoryStore,
) -> None:
    assert len(_rows(store)) == 0

    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    assert len(_rows(store)) == 1


def test_regression_create_memory_returns_correct_fields(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    assert memory.user_id == "user-1"
    assert memory.content == "favorite food is pizza"
    assert memory.memory_type == "semantic"
    assert memory.id
    assert memory.created_at
    assert memory.updated_at


def test_regression_update_memory_preserves_id(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    store.update_memory(
        memory_id=original.id,
        content="favorite color is blue",
    )

    updated = _row_by_id(store, original.id)
    assert updated["id"] == original.id


def test_regression_update_memory_row_count_unchanged(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1

    store.update_memory(
        memory_id=_rows(store)[0]["id"],
        content="favorite color is blue",
    )

    # Append-only: old row is inactive, new row is active — total rows = 2
    assert len(_rows(store)) == 2
    assert len(_active_rows(store)) == 1


def test_regression_update_memory_changes_content(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    store.update_memory(
        memory_id=original.id,
        content="favorite color is blue",
    )

    active = _active_rows(store)
    assert len(active) == 1
    assert active[0]["content"] == "favorite color is blue"
    assert active[0]["replaces_id"] == original.id


def test_regression_delete_memory_deactivates_row(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1

    store.delete_memory(memory_id=memory.id)

    # Soft delete: row stays but is_active flips to False
    assert len(_rows(store)) == 1
    row = _row_by_id(store, memory.id)
    assert row["is_active"] is False


def test_regression_delete_memory_sets_inactive(
    store: LanceDSPyMemoryStore,
) -> None:
    memory = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    store.delete_memory(memory_id=memory.id)

    row = _row_by_id(store, memory.id)
    assert row is not None
    assert row["is_active"] is False


def test_regression_search_memories_returns_results(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="hiking is my hobby",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
    )

    assert len(results) > 0
    result_contents = [r.content for r in results]
    assert any("hiking" in c for c in result_contents)


def test_regression_search_memories_respects_limit(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="hiking is my hobby",
        memory_type="semantic",
    )
    store.create_memory(
        user_id="user-1",
        content="enjoys outdoor activities",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
        limit=1,
    )

    assert len(results) == 1


def test_regression_search_memories_respects_memory_type_filter(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
        memory_type="preference",
    )

    assert len(results) >= 1
    for result in results:
        assert result.memory_type == "preference"


# ---------------------------------------------------------------------------
# Append-only semantics tests — verify NEW behavior
# ---------------------------------------------------------------------------


def test_append_only_update_creates_new_record(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC1: update_memory() creates a new record and supersedes the old one.

    After update: total row count +1, old record inactive, new record active
    with replaces_id pointing to the old record.
    """
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1
    assert len(_active_rows(store)) == 1

    store.update_memory(
        memory_id=original.id,
        content="favorite food is pepperoni pizza",
    )

    assert len(_rows(store)) == 2
    assert len(_active_rows(store)) == 1

    old = _row_by_id(store, original.id)
    assert old["is_active"] is False

    active = _active_rows(store)
    new = active[0]
    assert new["is_active"] is True
    assert new["content"] == "favorite food is pepperoni pizza"
    assert new["replaces_id"] == original.id


def test_append_only_search_excludes_superseded_records(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC2: search_memories() returns only active records, excluding superseded ones."""
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    store.update_memory(
        memory_id=original.id,
        content="favorite food is pepperoni pizza",
    )

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
    )

    result_ids = [r.id for r in results]
    assert original.id not in result_ids
    assert len(results) == 1
    assert results[0].content == "favorite food is pepperoni pizza"


def test_append_only_delete_soft_deletes(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC3: delete_memory() soft-deletes — row still exists, is_active=False,
    excluded from search."""
    memory = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1

    store.delete_memory(memory_id=memory.id)

    assert len(_rows(store)) == 1

    row = _row_by_id(store, memory.id)
    assert row["is_active"] is False

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
    )
    result_ids = [r.id for r in results]
    assert memory.id not in result_ids


def test_append_only_many_to_one_consolidation(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC4: A single new record with replaces_id='["id1","id2"]' can supersede
    multiple predecessors. Both old records become inactive."""
    import json

    mem_a = store.create_memory(
        user_id="user-1",
        content="car color is blue",
        memory_type="semantic",
    )
    mem_b = store.create_memory(
        user_id="user-1",
        content="car color is red",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 2
    assert len(_active_rows(store)) == 2

    new_row = store._build_memory_row(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        metadata=None,
    )
    new_row["replaces_id"] = json.dumps([mem_a.id, mem_b.id])
    store.table.add([new_row])

    store.delete_memory(memory_id=mem_a.id)
    store.delete_memory(memory_id=mem_b.id)

    assert len(_rows(store)) == 3
    assert len(_active_rows(store)) == 1

    assert _row_by_id(store, mem_a.id)["is_active"] is False
    assert _row_by_id(store, mem_b.id)["is_active"] is False

    active = _active_rows(store)
    assert active[0]["is_active"] is True
    replaces_ids = json.loads(active[0]["replaces_id"])
    assert set(replaces_ids) == {mem_a.id, mem_b.id}


def test_append_only_upsert_update_path_is_append_only(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC5: upsert_memory() "update" path is append-only — creates a new row
    and deactivates the old one."""
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="semantic",
        similarity_threshold=0.85,
        use_reconciler=False,
    )

    assert result.id != original.id
    assert len(_rows(store)) == 2
    assert len(_active_rows(store)) == 1

    assert _row_by_id(store, original.id)["is_active"] is False

    assert result.is_active is True
    assert result.replaces_id == original.id


def test_append_only_process_memories_delete_is_soft(
    store: LanceDSPyMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC6: process_memories() "delete" action performs a soft delete — row
    still exists with is_active=False."""
    from dspy_lancedb_memory.models import MemoryOperation

    memory = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )
    assert len(_rows(store)) == 1

    class StubDeleteExtractor:
        def __call__(self, *, messages):
            return dspy.Prediction(
                operations=[
                    MemoryOperation(
                        action="delete",
                        content="I love hiking",
                        search_query="hiking hobby",
                    )
                ]
            )

        def forward(self, messages):
            raise AssertionError("forward() should not be called directly")

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor", StubDeleteExtractor
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I no longer enjoy hiking"}],
        extract=True,
    )

    assert len(deleted) >= 1
    assert len(_rows(store)) == 1

    row = _row_by_id(store, memory.id)
    assert row["is_active"] is False

    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
    )
    result_ids = [r.id for r in results]
    assert memory.id not in result_ids


def test_create_memory_defaults(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC7: create_memory() returns Memory with is_active=True and
    replaces_id=None."""
    memory = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    assert memory.is_active is True
    assert memory.replaces_id is None


def test_append_only_history_chain_walkability(
    store: LanceDSPyMemoryStore,
) -> None:
    """AC8: Follow replaces_id backward through 2+ hops: A → B (replaces A)
    → C (replaces B). Starting from C, walk back to A."""
    mem_a = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    store.update_memory(
        memory_id=mem_a.id,
        content="hiking is my hobby",
    )
    active_after_b = _active_rows(store)
    assert len(active_after_b) == 1
    mem_b = SimpleNamespace(**active_after_b[0])
    assert mem_b.replaces_id == mem_a.id

    store.update_memory(
        memory_id=mem_b.id,
        content="enjoys outdoor activities",
    )
    active_after_c = _active_rows(store)
    assert len(active_after_c) == 1
    mem_c = SimpleNamespace(**active_after_c[0])
    assert mem_c.replaces_id == mem_b.id

    chain = [mem_c.id]
    current = _row_by_id(store, mem_c.id)
    while current.get("replaces_id"):
        chain.append(current["replaces_id"])
        current = _row_by_id(store, current["replaces_id"])

    assert chain == [mem_c.id, mem_b.id, mem_a.id]
    assert current["id"] == mem_a.id
    assert current["replaces_id"] is None


# ---------------------------------------------------------------------------
# Append-only upsert paths — verify update_memory() integration
# ---------------------------------------------------------------------------


def test_upsert_update_path_creates_new_record_old_becomes_inactive(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    initial_row_count = len(_rows(store))
    assert initial_row_count == 1

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="semantic",
        similarity_threshold=0.85,
        use_reconciler=False,
    )

    assert len(_rows(store)) == initial_row_count + 1
    assert len(_active_rows(store)) == 1

    old = _row_by_id(store, original.id)
    assert old["is_active"] is False

    assert result.id != original.id
    assert result.content == "favorite food is pepperoni pizza"
    new_row = _row_by_id(store, result.id)
    assert new_row["replaces_id"] == original.id


def test_upsert_keep_path_does_not_create_new_row(
    store: LanceDSPyMemoryStore,
) -> None:
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    initial_row_count = len(_rows(store))
    assert initial_row_count == 1

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        use_reconciler=False,
    )

    assert len(_rows(store)) == initial_row_count
    assert result.id == original.id


def test_upsert_create_path_has_replaces_id_none(
    store: LanceDSPyMemoryStore,
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite programming language is python",
        memory_type="semantic",
        use_reconciler=False,
    )

    assert result.content == "favorite programming language is python"
    new_row = _row_by_id(store, result.id)
    assert new_row["replaces_id"] is None


def test_search_memories_returns_empty_when_all_records_inactive(
    store: LanceDSPyMemoryStore,
) -> None:
    mem = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=mem.id)

    assert _active_rows(store) == []

    results = store.search_memories(
        user_id="user-1",
        query="what food do I like",
    )
    assert results == []


def test_update_memory_nonexistent_id_is_noop(
    store: LanceDSPyMemoryStore,
) -> None:
    import uuid

    before = len(_rows(store))
    store.update_memory(
        memory_id=str(uuid.uuid4()),
        content="favorite food is pizza",
    )
    assert len(_rows(store)) == before


def test_update_memory_on_inactive_record_is_noop(
    store: LanceDSPyMemoryStore,
) -> None:
    mem = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=mem.id)
    rows_after_delete = len(_rows(store))
    assert rows_after_delete == 1

    store.update_memory(
        memory_id=mem.id,
        content="favorite food is pepperoni pizza",
    )
    assert len(_rows(store)) == rows_after_delete


def test_delete_memory_on_already_inactive_is_idempotent(
    store: LanceDSPyMemoryStore,
) -> None:
    mem = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=mem.id)

    store.delete_memory(memory_id=mem.id)

    assert len(_rows(store)) == 1
    row = _row_by_id(store, mem.id)
    assert row["is_active"] is False


# ---------------------------------------------------------------------------
# Skip-threshold tests — near-duplicate suppression
# ---------------------------------------------------------------------------


def test_upsert_semantic_skip_threshold_skips_near_duplicate(
    store: LanceDSPyMemoryStore,
) -> None:
    """When cosine similarity >= skip_threshold (0.92) and the new content is
    not genuinely richer, upsert should skip — returning the existing memory
    without creating a duplicate."""
    original = store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="hiking is enjoyable",
        memory_type="semantic",
        use_reconciler=False,
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    assert _row_by_id(store, original.id)["content"] == "I love hiking"


def test_upsert_semantic_skip_threshold_allows_richer_update(
    store: LanceDSPyMemoryStore,
) -> None:
    """When cosine similarity >= skip_threshold but the new content IS richer
    (adds detail), upsert should still update — the skip gate only blocks
    near-duplicates that aren't improvements."""
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

    assert result.id != original.id
    assert len(_active_rows(store)) == 1
    assert result.content == "favorite food is pepperoni pizza"
    assert _row_by_id(store, original.id)["is_active"] is False


def test_upsert_nonsemantic_skip_threshold_skips_near_duplicate(
    store: LanceDSPyMemoryStore,
) -> None:
    """Non-semantic path: when cosine similarity >= skip_threshold, skip
    the write even though the content strings differ."""
    original = store.create_memory(
        user_id="user-1",
        content="car color is blue",
        memory_type="preference",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="vehicle color blue",
        memory_type="preference",
        use_reconciler=False,
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
    assert _row_by_id(store, original.id)["content"] == "car color is blue"


def test_upsert_nonsemantic_skips_when_above_similarity_threshold(
    store: LanceDSPyMemoryStore,
) -> None:
    """Non-semantic path: when cosine similarity >= skip_threshold (0.85),
    the memory is considered close enough to skip — no update, no create."""
    original = store.create_memory(
        user_id="user-1",
        content="car color is blue",
        memory_type="preference",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="my car is blue",
        memory_type="preference",
        use_reconciler=False,
    )

    assert result.id == original.id
    assert len(_rows(store)) == 1
