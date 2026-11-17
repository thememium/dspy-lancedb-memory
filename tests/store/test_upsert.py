from __future__ import annotations

from types import SimpleNamespace

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
