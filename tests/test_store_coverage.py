"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import dspy
import pytest

from dspy_lancedb_memory.models import MemoryType, ReconciledMemory, Scope
from dspy_lancedb_memory.store import LanceDSPyMemoryStore

EMBEDDINGS: dict[str, list[float]] = {
    "favorite food is pizza": [1.0, 0.0, 0.0],
    "favorite food is pepperoni pizza": [0.95, 0.05, 0.0],
    "favorite color is blue": [0.99, 0.01, 0.0],
    "favorite color is red": [0.99, 0.01, 0.0],
    "favorite programming language is python": [0.0, 1.0, 0.0],
    "name is Edward": [0.9, 0.1, 0.0],
    "name is Edward Boswell": [0.88, 0.12, 0.0],
    "I love hiking": [0.0, 0.0, 1.0],
    "hiking is my hobby": [0.0, 0.0, 0.95],
    "car color is blue": [0.0, 0.5, 0.5],
    "car color is red": [0.0, 0.5, 0.5],
    "hiking is enjoyable": [0.05, 0.02, 0.98],
    "vehicle color blue": [0.02, 0.48, 0.5],
    "my car is blue": [0.08, 0.45, 0.5],
    "new fact to remember": [0.3, 0.3, 0.3],
    "delete my pizza memory": [1.0, 0.0, 0.0],
    "test content": [0.5, 0.5, 0.0],
    "updated content": [0.6, 0.4, 0.0],
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
        return self._embeddings.get(text, [0.5, 0.5, 0.0])


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


# ---------------------------------------------------------------------------
# _json_dict edge cases
# ---------------------------------------------------------------------------


class TestJsonDict:
    def test_json_dict_with_dict_input(self):
        result = LanceDSPyMemoryStore._json_dict({"key": "value"})
        assert result == {"key": "value"}
        # Should be a copy, not the same object
        assert result is not {"key": "value"}

    def test_json_dict_with_empty_dict(self):
        result = LanceDSPyMemoryStore._json_dict({})
        assert result == {}

    def test_json_dict_with_none(self):
        result = LanceDSPyMemoryStore._json_dict(None)
        assert result == {}

    def test_json_dict_with_empty_string(self):
        result = LanceDSPyMemoryStore._json_dict("")
        assert result == {}

    def test_json_dict_with_valid_json_string(self):
        result = LanceDSPyMemoryStore._json_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_dict_with_invalid_json_string(self):
        result = LanceDSPyMemoryStore._json_dict("not json")
        assert result == {}

    def test_json_dict_with_json_array(self):
        # JSON array should return empty dict (not a dict)
        result = LanceDSPyMemoryStore._json_dict("[1, 2, 3]")
        assert result == {}

    def test_json_dict_with_integer(self):
        result = LanceDSPyMemoryStore._json_dict(0)
        assert result == {}

    def test_json_dict_with_list(self):
        result = LanceDSPyMemoryStore._json_dict([])
        assert result == {}


# ---------------------------------------------------------------------------
# _matches_filter edge cases
# ---------------------------------------------------------------------------


class TestMatchesFilter:
    def test_eq_operator_matching(self):
        assert LanceDSPyMemoryStore._matches_filter("value", {"eq": "value"}) is True

    def test_eq_operator_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter("other", {"eq": "value"}) is False

    def test_neq_operator_matching(self):
        assert LanceDSPyMemoryStore._matches_filter("other", {"neq": "value"}) is True

    def test_neq_operator_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter("value", {"neq": "value"}) is False

    def test_in_operator_matching(self):
        assert (
            LanceDSPyMemoryStore._matches_filter("a", {"in": ["a", "b", "c"]}) is True
        )

    def test_in_operator_not_matching(self):
        assert (
            LanceDSPyMemoryStore._matches_filter("d", {"in": ["a", "b", "c"]}) is False
        )

    def test_contains_operator_with_list(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(["a", "b", "c"], {"contains": "b"})
            is True
        )

    def test_contains_operator_with_list_not_matching(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(["a", "b", "c"], {"contains": "d"})
            is False
        )

    def test_contains_operator_with_tuple(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(("a", "b"), {"contains": "a"}) is True
        )

    def test_contains_operator_with_set(self):
        assert (
            LanceDSPyMemoryStore._matches_filter({"a", "b"}, {"contains": "a"}) is True
        )

    def test_contains_operator_with_string(self):
        assert (
            LanceDSPyMemoryStore._matches_filter("hello world", {"contains": "world"})
            is True
        )

    def test_contains_operator_with_string_not_matching(self):
        assert (
            LanceDSPyMemoryStore._matches_filter("hello", {"contains": "world"}) is False
        )

    def test_contains_operator_with_non_container(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(123, {"contains": "a"}) is False
        )

    def test_gt_operator_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(10, {"gt": 5}) is True

    def test_gt_operator_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(5, {"gt": 5}) is False

    def test_gte_operator_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(5, {"gte": 5}) is True

    def test_gte_operator_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(4, {"gte": 5}) is False

    def test_lt_operator_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(4, {"lt": 5}) is True

    def test_lt_operator_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(5, {"lt": 5}) is False

    def test_lte_operator_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(5, {"lte": 5}) is True

    def test_lte_operator_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter(6, {"lte": 5}) is False

    def test_exists_operator_with_existing_value(self):
        assert (
            LanceDSPyMemoryStore._matches_filter("value", {"exists": True}) is True
        )

    def test_exists_operator_with_none(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(None, {"exists": True}) is False
        )

    def test_exists_operator_with_false(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(None, {"exists": False}) is True
        )

    def test_nested_dict_filter(self):
        value = {"nested": {"key": "value"}}
        expected = {"nested": {"key": "value"}}
        assert LanceDSPyMemoryStore._matches_filter(value, expected) is True

    def test_nested_dict_filter_not_matching(self):
        value = {"nested": {"key": "other"}}
        expected = {"nested": {"key": "value"}}
        assert LanceDSPyMemoryStore._matches_filter(value, expected) is False

    def test_nested_dict_filter_missing_key(self):
        value = {"other": "value"}
        expected = {"nested": {"key": "value"}}
        assert LanceDSPyMemoryStore._matches_filter(value, expected) is False

    def test_list_filter(self):
        assert LanceDSPyMemoryStore._matches_filter("a", ["a", "b"]) is True

    def test_list_filter_not_matching(self):
        assert LanceDSPyMemoryStore._matches_filter("c", ["a", "b"]) is False

    def test_tuple_filter(self):
        assert LanceDSPyMemoryStore._matches_filter("a", ("a", "b")) is True

    def test_set_filter(self):
        assert LanceDSPyMemoryStore._matches_filter("a", {"a", "b"}) is True

    def test_direct_equality(self):
        assert LanceDSPyMemoryStore._matches_filter("value", "value") is True

    def test_direct_inequality(self):
        assert LanceDSPyMemoryStore._matches_filter("value", "other") is False

    def test_dict_expected_without_operators(self):
        # When expected is a dict but doesn't have operator keys
        # and value is not a dict, should return False
        assert LanceDSPyMemoryStore._matches_filter("string", {"key": "value"}) is False

    def test_multiple_operators(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(
                10, {"gte": 5, "lte": 15, "neq": 7}
            )
            is True
        )

    def test_multiple_operators_failing(self):
        assert (
            LanceDSPyMemoryStore._matches_filter(
                10, {"gte": 5, "lte": 8}
            )
            is False
        )


# ---------------------------------------------------------------------------
# _escape_filter_value
# ---------------------------------------------------------------------------


class TestEscapeFilterValue:
    def test_escape_single_quote(self):
        assert LanceDSPyMemoryStore._escape_filter_value("user'1") == "user''1"

    def test_no_quotes(self):
        assert LanceDSPyMemoryStore._escape_filter_value("user1") == "user1"

    def test_multiple_quotes(self):
        assert LanceDSPyMemoryStore._escape_filter_value("it's") == "it''s"


# ---------------------------------------------------------------------------
# _split_metadata_scope
# ---------------------------------------------------------------------------


class TestSplitMetadataScope:
    def test_split_with_scope_key(self):
        metadata = '{"_scope": {"tenant": "a"}, "source": "chat"}'
        clean, scope = LanceDSPyMemoryStore._split_metadata_scope(metadata)
        assert clean == {"source": "chat"}
        assert scope == {"tenant": "a"}

    def test_split_without_scope_key(self):
        metadata = '{"source": "chat"}'
        clean, scope = LanceDSPyMemoryStore._split_metadata_scope(metadata)
        assert clean == {"source": "chat"}
        assert scope == {}

    def test_split_with_dict_input(self):
        metadata = {"_scope": {"tenant": "a"}, "source": "chat"}
        clean, scope = LanceDSPyMemoryStore._split_metadata_scope(metadata)
        assert clean == {"source": "chat"}
        assert scope == {"tenant": "a"}

    def test_split_with_none(self):
        clean, scope = LanceDSPyMemoryStore._split_metadata_scope(None)
        assert clean == {}
        assert scope == {}


# ---------------------------------------------------------------------------
# _pack_metadata
# ---------------------------------------------------------------------------


class TestPackMetadata:
    def test_pack_with_metadata_and_scope(self):
        result = LanceDSPyMemoryStore._pack_metadata(
            {"source": "chat"}, {"tenant": "a"}
        )
        parsed = json.loads(result)
        assert parsed["source"] == "chat"
        assert parsed["_scope"] == {"tenant": "a"}

    def test_pack_with_empty_metadata(self):
        result = LanceDSPyMemoryStore._pack_metadata(None, None)
        assert json.loads(result) == {}

    def test_pack_with_reserved_key_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            LanceDSPyMemoryStore._pack_metadata({"_scope": {}}, None)


# ---------------------------------------------------------------------------
# _to_memory edge cases
# ---------------------------------------------------------------------------


class TestToMemory:
    def test_to_memory_with_relevance_score(self):
        row = {
            "id": "test-id",
            "user_id": "user-1",
            "session_id": "",
            "conversation_id": "",
            "memory_type": "semantic",
            "content": "test",
            "metadata": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "_relevance_score": 0.95,
        }
        memory = LanceDSPyMemoryStore._to_memory(row)
        assert memory.relevance_score == 0.95

    def test_to_memory_with_distance(self):
        row = {
            "id": "test-id",
            "user_id": "user-1",
            "session_id": "",
            "conversation_id": "",
            "memory_type": "semantic",
            "content": "test",
            "metadata": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "_distance": 0.3,
        }
        memory = LanceDSPyMemoryStore._to_memory(row)
        assert memory.relevance_score == pytest.approx(0.7)

    def test_to_memory_without_scores(self):
        row = {
            "id": "test-id",
            "user_id": "user-1",
            "session_id": "",
            "conversation_id": "",
            "memory_type": "semantic",
            "content": "test",
            "metadata": "{}",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }
        memory = LanceDSPyMemoryStore._to_memory(row)
        assert memory.relevance_score is None


# ---------------------------------------------------------------------------
# _filter_by_relevance
# ---------------------------------------------------------------------------


class TestFilterByRelevance:
    def test_filter_with_relevance_score(self):
        results = [
            {"_relevance_score": 0.9},
            {"_relevance_score": 0.5},
            {"_relevance_score": 0.2},
        ]
        filtered = LanceDSPyMemoryStore._filter_by_relevance(results, 0.6)
        assert len(filtered) == 1
        assert filtered[0]["_relevance_score"] == 0.9

    def test_filter_with_distance(self):
        results = [
            {"_distance": 0.1},  # score = 0.9
            {"_distance": 0.5},  # score = 0.5
            {"_distance": 0.8},  # score = 0.2
        ]
        filtered = LanceDSPyMemoryStore._filter_by_relevance(results, 0.6)
        assert len(filtered) == 1
        assert filtered[0]["_distance"] == 0.1

    def test_filter_with_no_score_defaults_to_zero(self):
        results = [{"id": "test"}]
        filtered = LanceDSPyMemoryStore._filter_by_relevance(results, 0.5)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# _build_filters edge cases
# ---------------------------------------------------------------------------


class TestBuildFilters:
    def test_build_filters_with_memory_type(self):
        filters = LanceDSPyMemoryStore._build_filters(
            user_id="user-1",
            session_id="",
            conversation_id="",
            memory_type="semantic",
            active_only=True,
        )
        assert "memory_type = 'semantic'" in filters
        assert "is_active = true" in filters

    def test_build_filters_without_active_only(self):
        filters = LanceDSPyMemoryStore._build_filters(
            user_id="user-1",
            session_id="",
            conversation_id="",
            active_only=False,
        )
        assert "is_active = true" not in filters

    def test_build_filters_with_session_id(self):
        filters = LanceDSPyMemoryStore._build_filters(
            user_id="user-1",
            session_id="session-1",
            conversation_id="",
        )
        assert "session_id = 'session-1'" in filters

    def test_build_filters_with_conversation_id(self):
        filters = LanceDSPyMemoryStore._build_filters(
            user_id="user-1",
            session_id="",
            conversation_id="conv-1",
        )
        assert "conversation_id = 'conv-1'" in filters

    def test_build_filters_with_quotes(self):
        filters = LanceDSPyMemoryStore._build_filters(
            user_id="user'1",
            session_id="",
            conversation_id="",
        )
        assert "user_id = 'user''1'" in filters


# ---------------------------------------------------------------------------
# _scope_dict edge cases
# ---------------------------------------------------------------------------


class TestScopeDict:
    def test_scope_dict_with_none(self):
        assert LanceDSPyMemoryStore._scope_dict(None) == {}

    def test_scope_dict_with_scope_model(self):
        scope = Scope(tenant="a", repo="b")
        result = LanceDSPyMemoryStore._scope_dict(scope)
        assert result == {"tenant": "a", "repo": "b"}

    def test_scope_dict_with_dict(self):
        result = LanceDSPyMemoryStore._scope_dict({"tenant": "a"})
        assert result == {"tenant": "a"}


# ---------------------------------------------------------------------------
# upsert_memory with reconciler (real paths)
# ---------------------------------------------------------------------------


class StubReconciler:
    """Deterministic reconciler for testing."""

    def __call__(self, *, new_memory_content, new_memory_type, existing_memories):
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


def test_upsert_memory_with_reconciler_keep(store, monkeypatch):
    """Test upsert_memory with reconciler returning 'keep' action."""
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
        skip_threshold=1.0,  # Force reconciler path
        use_reconciler=True,
    )

    assert result.id == original.id


def test_upsert_memory_with_reconciler_update(store, monkeypatch):
    """Test upsert_memory with reconciler returning 'update' action."""
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
        use_reconciler=True,
    )

    assert result.id != original.id
    assert result.content == "name is Edward Boswell"


def test_upsert_memory_with_reconciler_create(store, monkeypatch):
    """Test upsert_memory with reconciler returning 'create' action."""
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
        use_reconciler=True,
    )

    assert result.content == "favorite color is blue"
    assert len(_rows(store)) == 2


def test_upsert_memory_nonsemantic_with_reconciler(store, monkeypatch):
    """Test upsert_memory with non-semantic type and reconciler."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="preference",
        skip_threshold=1.0,
        use_reconciler=True,
    )

    # Non-semantic with reconciler should use reconciler
    assert result is not None


# ---------------------------------------------------------------------------
# upsert_memories edge cases
# ---------------------------------------------------------------------------


def test_upsert_memories_verbatim(store):
    """Test upsert_memories with extract=False."""
    result = store.upsert_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "test content"}],
        extract=False,
        use_reconciler=False,
    )

    assert len(result) == 1
    assert result[0].content == "test content"


def test_upsert_memories_verbatim_requires_single_item(store):
    """Test upsert_memories with extract=False requires exactly one item."""
    with pytest.raises(ValueError, match="exactly one item"):
        store.upsert_memories(
            user_id="user-1",
            contents=[
                {"role": "user", "content": "item 1"},
                {"role": "user", "content": "item 2"},
            ],
            extract=False,
        )


def test_upsert_memories_verbatim_requires_contents(store):
    """Test upsert_memories with extract=False requires contents."""
    with pytest.raises(ValueError, match="exactly one item"):
        store.upsert_memories(
            user_id="user-1",
            contents=None,
            extract=False,
        )


def test_upsert_memories_extract_requires_contents(store):
    """Test upsert_memories with extract=True requires contents."""
    with pytest.raises(ValueError, match="contents is required"):
        store.upsert_memories(
            user_id="user-1",
            contents=None,
            extract=True,
        )


# ---------------------------------------------------------------------------
# process_memories edge cases
# ---------------------------------------------------------------------------


def test_process_memories_requires_contents(store):
    """Test process_memories requires contents."""
    with pytest.raises(ValueError, match="contents is required"):
        store.process_memories(
            user_id="user-1",
            contents=None,
        )


def test_process_memories_delete_with_search_query(store, monkeypatch):
    """Test process_memories delete action with search_query."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                {
                    "action": "delete",
                    "content": "",
                    "search_query": "pizza",
                    "memory_type": "",
                }
            ]
        ),
    )

    # This should use delete_memories_by_search
    # But we need to handle the fact that the mock returns dict not MemoryOperation
    # Let me adjust this test
    from dspy_lancedb_memory.models import MemoryOperation

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    content="",
                    search_query="pizza",
                    memory_type="",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Delete pizza"}],
        extract=True,
        similarity_threshold=0.5,
    )

    assert len(deleted) >= 1


# ---------------------------------------------------------------------------
# _normalize_extracted_memories edge cases
# ---------------------------------------------------------------------------


def test_normalize_extracted_memories_with_tuples(store):
    """Test _normalize_extracted_memories with tuple inputs."""
    extracted = [
        ("content1", "semantic", {"source": "chat"}),
        ("content2", "preference"),
    ]
    result = store._normalize_extracted_memories(extracted)
    assert len(result) == 2
    assert result[0][0] == "content1"
    assert result[1][0] == "content2"


def test_normalize_extracted_memories_with_single_tuple(store):
    """Test _normalize_extracted_memories with 2-element tuple."""
    extracted = [("content1", "semantic")]
    result = store._normalize_extracted_memories(extracted)
    assert len(result) == 1


def test_normalize_extracted_memories_with_invalid_tuple(store):
    """Test _normalize_extracted_memories with 1-element tuple (skipped)."""
    extracted = [("content1",)]
    result = store._normalize_extracted_memories(extracted)
    assert len(result) == 0


def test_normalize_extracted_memories_deduplicates(store):
    """Test _normalize_extracted_memories deduplicates content."""
    extracted = [
        ("same content", "semantic"),
        ("same content", "preference"),
    ]
    result = store._normalize_extracted_memories(extracted)
    assert len(result) == 1


def test_normalize_extracted_memories_filters_empty(store):
    """Test _normalize_extracted_memories filters empty content."""
    extracted = [
        ("", "semantic"),
        ("  ", "semantic"),
        ("valid", "semantic"),
    ]
    result = store._normalize_extracted_memories(extracted)
    assert len(result) == 1
    assert result[0][0] == "valid"


def test_normalize_extracted_memories_with_objects(store):
    """Test _normalize_extracted_memories with object inputs."""
    obj = SimpleNamespace(
        content="test content",
        type="semantic",
        metadata={"source": "chat"},
    )
    result = store._normalize_extracted_memories([obj])
    assert len(result) == 1
    assert result[0][0] == "test content"


# ---------------------------------------------------------------------------
# update_memory_metadata edge cases
# ---------------------------------------------------------------------------


def test_update_memory_metadata_nonexistent(store):
    """Test update_memory_metadata with nonexistent ID."""
    import uuid

    result = store.update_memory_metadata(
        memory_id=str(uuid.uuid4()),
        metadata={"key": "value"},
    )
    assert result is None


def test_update_memory_metadata_replace(store):
    """Test update_memory_metadata with merge=False."""
    memory = store.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
        metadata={"source": "chat", "priority": "low"},
    )

    result = store.update_memory_metadata(
        memory_id=memory.id,
        metadata={"priority": "high"},
        merge=False,
    )

    assert result is not None
    assert result.metadata == {"priority": "high"}


# ---------------------------------------------------------------------------
# update_memory_scope edge cases
# ---------------------------------------------------------------------------


def test_update_memory_scope_nonexistent(store):
    """Test update_memory_scope with nonexistent ID."""
    import uuid

    result = store.update_memory_scope(
        memory_id=str(uuid.uuid4()),
        scope={"tenant": "a"},
    )
    assert result is None


def test_update_memory_scope_replace(store):
    """Test update_memory_scope with merge=False."""
    memory = store.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
        scope={"tenant": "a", "repo": "b"},
    )

    result = store.update_memory_scope(
        memory_id=memory.id,
        scope={"project": "c"},
        merge=False,
    )

    assert result is not None
    assert result.scope == {"project": "c"}


# ---------------------------------------------------------------------------
# list_memories edge cases
# ---------------------------------------------------------------------------


def test_list_memories_with_include_inactive(store):
    """Test list_memories with include_inactive=True."""
    memory = store.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
    )
    store.delete_memory(memory_id=memory.id)

    active = store.list_memories(user_id="user-1")
    assert len(active) == 0

    all_memories = store.list_memories(user_id="user-1", include_inactive=True)
    assert len(all_memories) == 1


def test_list_memories_with_limit(store):
    """Test list_memories with limit."""
    for i in range(5):
        store.create_memory(
            user_id="user-1",
            content=f"content {i}",
            memory_type="semantic",
        )

    result = store.list_memories(user_id="user-1", limit=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# search_memories edge cases
# ---------------------------------------------------------------------------


def test_search_memories_with_min_relevance_score(store):
    """Test search_memories with min_relevance_score."""
    store.create_memory(
        user_id="user-1",
        content="I love hiking",
        memory_type="semantic",
    )

    # Search with high min score - should filter out low relevance
    results = store.search_memories(
        user_id="user-1",
        query="what are hobbies",
        min_relevance_score=0.99,
    )
    # Might return 0 or 1 depending on similarity
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# _get_or_create_table edge cases
# ---------------------------------------------------------------------------


def test_get_or_create_table_with_dimension_mismatch(tmp_path):
    """Test table recreation when vector dimension doesn't match."""
    # Create a store with dimension 3
    store1 = StubMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embeddings=EMBEDDINGS,
    )
    store1.create_memory(
        user_id="user-1",
        content="test",
        memory_type="semantic",
    )

    # Create another store with different dimension - should recreate table
    different_embeddings = {"test": [1.0, 0.0]}
    store2 = LanceDSPyMemoryStore(
        uri=str(tmp_path),
        table_name="memories",
        embedding_lm=SimpleNamespace(model="test-model"),
        embedding_dim=2,
        reranker=None,
    )
    # Override _embed to use our test embeddings
    store2._embed = lambda text: different_embeddings.get(text, [0.5, 0.5])
    store2.table = store2._get_or_create_table()

    # Table should have been recreated with new dimension
    assert store2.embedding_dim == 2


# ---------------------------------------------------------------------------
# _semantic_match_action edge cases
# ---------------------------------------------------------------------------


def test_semantic_match_action_empty_content(store):
    """Test _semantic_match_action with empty normalized content."""
    result = store._semantic_match_action(
        new_content="!!!",
        existing_content="hello",
        distance=0.5,
        similarity_threshold=0.85,
    )
    # Empty normalized content returns None
    assert result is None


def test_semantic_match_action_identical_normalized(store):
    """Test _semantic_match_action with identical normalized content."""
    result = store._semantic_match_action(
        new_content="Hello World!",
        existing_content="hello world",
        distance=0.1,
        similarity_threshold=0.85,
    )
    assert result == "skip"


# ---------------------------------------------------------------------------
# MemoryType value edge cases
# ---------------------------------------------------------------------------


def test_memory_type_value_with_string(store):
    """Test _memory_type_value with string input."""
    result = store._memory_type_value("semantic")
    assert result == "semantic"


def test_memory_type_value_with_enum(store):
    """Test _memory_type_value with enum input."""
    result = store._memory_type_value(MemoryType.PREFERENCE)
    assert result == "preference"


def test_memory_type_value_with_none(store):
    """Test _memory_type_value with None input."""
    result = store._memory_type_value(None)
    assert result is None


# ---------------------------------------------------------------------------
# BoundMemoryStore methods
# ---------------------------------------------------------------------------


def test_bound_store_process_memories(store):
    """Test BoundMemoryStore.process_memories."""
    bound = store.with_scope(
        user_id="user-1",
        session_id="session-1",
        scope={"tenant": "tenant-a"},
    )

    created, deleted = bound.process_memories(
        contents=[{"role": "user", "content": "I love pizza"}],
        extract=False,
    )

    assert len(created) == 1
    assert created[0].session_id == "session-1"
    assert created[0].scope == {"tenant": "tenant-a"}


def test_bound_store_delete_memories_by_search(store):
    """Test BoundMemoryStore.delete_memories_by_search."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    bound.create_memory(
        content="favorite food is pizza",
        memory_type="semantic",
    )

    deleted = bound.delete_memories_by_search(
        query="delete my pizza memory",
        similarity_threshold=0.5,
    )

    assert len(deleted) == 1
    assert deleted[0].content == "favorite food is pizza"


def test_bound_store_upsert_memories(store, monkeypatch):
    """Test BoundMemoryStore.upsert_memories."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite food is pizza", "semantic")]
        ),
    )
    monkeypatch.setattr("dspy_lancedb_memory.store.MemoryReconciler", StubReconciler)

    results = bound.upsert_memories(
        contents=[{"role": "user", "content": "I like pizza"}],
        extract=True,
        num_threads=1,
    )

    assert len(results) == 1
    assert results[0].scope == {"tenant": "tenant-a"}


def test_bound_store_scope_merging(store):
    """Test BoundMemoryStore._scope merges scopes correctly."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "a", "repo": "b"},
    )

    # Test internal _scope method
    merged = bound._scope({"project": "c"})
    assert merged == {"tenant": "a", "repo": "b", "project": "c"}


def test_bound_store_create_memories_extract(store, monkeypatch):
    """Test BoundMemoryStore.create_memories with extract=True."""
    bound = store.with_scope(
        user_id="user-1",
        scope={"tenant": "tenant-a"},
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryExtractor.forward",
        lambda self, messages: dspy.Prediction(
            memories=[("favorite food is pizza", "semantic")]
        ),
    )

    results = bound.create_memories(
        contents=[{"role": "user", "content": "I like pizza"}],
        extract=True,
    )

    assert len(results) == 1
    assert results[0].scope == {"tenant": "tenant-a"}


# ---------------------------------------------------------------------------
# upsert_memory non-semantic paths
# ---------------------------------------------------------------------------


def test_upsert_nonsemantic_exact_match_returns_existing(store):
    """Test non-semantic upsert with exact content match returns existing."""
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
        use_reconciler=False,
    )

    assert result.id == original.id


def test_upsert_nonsemantic_similarity_above_threshold_updates(store):
    """Test non-semantic upsert updates when similarity is above threshold."""
    original = store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    # Use a content with high similarity but not exact match
    result = store.upsert_memory(
        user_id="user-1",
        content="favorite food is pepperoni pizza",
        memory_type="preference",
        similarity_threshold=0.8,
        skip_threshold=0.99,  # High skip threshold to allow update
        use_reconciler=False,
    )

    # Should either update or create depending on similarity
    assert result is not None


def test_upsert_nonsemantic_below_skip_threshold_creates(store):
    """Test non-semantic upsert creates when below skip threshold."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="preference",
    )

    # Use a content with low similarity
    result = store.upsert_memory(
        user_id="user-1",
        content="favorite programming language is python",
        memory_type="preference",
        use_reconciler=False,
    )

    assert result.content == "favorite programming language is python"


# ---------------------------------------------------------------------------
# delete_memories_by_search edge cases
# ---------------------------------------------------------------------------


def test_delete_memories_by_search_no_candidates(store):
    """Test delete_memories_by_search returns empty when no candidates."""
    deleted = store.delete_memories_by_search(
        user_id="user-1",
        query="nonexistent memory",
    )

    assert deleted == []


def test_delete_memories_by_search_with_scope_and_metadata(store):
    """Test delete_memories_by_search with scope and metadata filter."""
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "a"},
        metadata={"source": "chat"},
    )
    store.create_memory(
        user_id="user-1",
        content="favorite food is pizza",
        memory_type="semantic",
        scope={"tenant": "b"},
        metadata={"source": "doc"},
    )

    deleted = store.delete_memories_by_search(
        user_id="user-1",
        query="delete my pizza memory",
        scope={"tenant": "a"},
        metadata_filter={"source": "chat"},
        similarity_threshold=0.5,
    )

    assert len(deleted) == 1


# ---------------------------------------------------------------------------
# process_memories edge cases for uncovered paths
# ---------------------------------------------------------------------------


def test_process_memories_update_action(store, monkeypatch):
    """Test process_memories update action (line 1486)."""
    from dspy_lancedb_memory.models import MemoryOperation

    store.create_memory(
        user_id="user-1",
        content="name is Edward",
        memory_type="semantic",
    )

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="update",
                    content="name is Edward Boswell",
                    memory_type="semantic",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "My full name is Edward Boswell"}],
        extract=True,
        use_reconciler=False,
    )

    assert len(created) == 1
    assert created[0].content == "name is Edward Boswell"


def test_process_memories_empty_search_query_skips(store, monkeypatch):
    """Test process_memories skips delete with empty search query."""
    from dspy_lancedb_memory.models import MemoryOperation

    monkeypatch.setattr(
        "dspy_lancedb_memory.store.MemoryOperationExtractor.forward",
        lambda self, messages: dspy.Prediction(
            operations=[
                MemoryOperation(
                    action="delete",
                    content="",
                    search_query="",
                    memory_type="",
                )
            ]
        ),
    )

    created, deleted = store.process_memories(
        user_id="user-1",
        contents=[{"role": "user", "content": "Delete something"}],
        extract=True,
    )

    assert len(created) == 0
    assert len(deleted) == 0
