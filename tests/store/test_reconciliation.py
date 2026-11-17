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


def test_upsert_memories_extract_path_reuses_semantic_upsert_decision(
    store: LanceDSPyMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
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

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.read.MemoryExtractor", DummyExtractor
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryExtractor", DummyExtractor
    )

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

    # Custom reconciler that returns "keep" for semantically similar content
    class KeepReconciler:
        def __init__(self):
            pass

        def __call__(self, *, new_memory_content, new_memory_type, existing_memories):
            from dspy_lancedb_memory.models import ReconciledMemory

            # Return "keep" for the first existing memory (simulating semantic match)
            if existing_memories:
                return dspy.Prediction(
                    reconciled=ReconciledMemory(
                        action="keep",
                        memory_id=existing_memories[0]["id"],
                        final_content=existing_memories[0]["content"],
                        final_type=existing_memories[0]["type"],
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

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", KeepReconciler
    )

    # Use content with similar but not identical embedding to avoid skip_threshold gate
    # "my name is edward" has embedding [0.85, 0.15, 0.0] vs "name is Edward" [0.9, 0.1, 0.0]
    # This gives similarity ~0.93 which is < skip_threshold=1.0, so reconciler is called
    result = store.upsert_memory(
        user_id="user-1",
        content="my name is edward",
        memory_type="semantic",
        skip_threshold=1.0,  # Force reconciler path by setting high skip_threshold
    )

    # KeepReconciler returns "keep" for the first existing memory
    assert result.id == original.id
    assert result.content == "name is Edward"
    assert len(_rows(store)) == 1


def test_reconciler_updates_refinement(store, monkeypatch):
    original = store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("hiking is enjoyable", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("favorite food is pizza", "semantic"),
                ("favorite color is blue", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like pizza and blue."}],
        extract=True,
        num_threads=2,
    )

    assert len(results) == 2


def test_parallel_with_multiple_memories(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("favorite food is pizza", "semantic"),
                ("favorite color is blue", "semantic"),
                ("favorite programming language is python", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite color is blue", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("name is Edward", "semantic"),
                ("favorite color is blue", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward Boswell", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[
                ("favorite food is pizza", "semantic"),
                ("favorite food is pizza", "semantic"),
                ("favorite color is blue", "semantic"),
            ]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

    results = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "I like pizza and blue."}],
        extract=True,
        num_threads=4,
    )

    assert len(results) == 2


def test_parallel_empty_extraction(store, monkeypatch):
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(memories=[]),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
        "dspy_lancedb_memory.store.upsert.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("name is Edward", "semantic")]
        ),
    )
    monkeypatch.setattr(
        "dspy_lancedb_memory.store.upsert.MemoryReconciler", StubReconciler
    )

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
