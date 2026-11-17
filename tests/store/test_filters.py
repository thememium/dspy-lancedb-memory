"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

from dspy_lancedb_memory.store.filters import (
    build_filters,
    escape_filter_value,
    matches_filter,
)

# ---------------------------------------------------------------------------
# _matches_filter edge cases
# ---------------------------------------------------------------------------


class TestMatchesFilter:
    def test_eq_operator_matching(self):
        assert matches_filter("value", {"eq": "value"}) is True

    def test_eq_operator_not_matching(self):
        assert matches_filter("other", {"eq": "value"}) is False

    def test_neq_operator_matching(self):
        assert matches_filter("other", {"neq": "value"}) is True

    def test_neq_operator_not_matching(self):
        assert matches_filter("value", {"neq": "value"}) is False

    def test_in_operator_matching(self):
        assert matches_filter("a", {"in": ["a", "b", "c"]}) is True

    def test_in_operator_not_matching(self):
        assert matches_filter("d", {"in": ["a", "b", "c"]}) is False

    def test_contains_operator_with_list(self):
        assert matches_filter(["a", "b", "c"], {"contains": "b"}) is True

    def test_contains_operator_with_list_not_matching(self):
        assert matches_filter(["a", "b", "c"], {"contains": "d"}) is False

    def test_contains_operator_with_tuple(self):
        assert matches_filter(("a", "b"), {"contains": "a"}) is True

    def test_contains_operator_with_set(self):
        assert matches_filter({"a", "b"}, {"contains": "a"}) is True

    def test_contains_operator_with_string(self):
        assert matches_filter("hello world", {"contains": "world"}) is True

    def test_contains_operator_with_string_not_matching(self):
        assert matches_filter("hello", {"contains": "world"}) is False

    def test_contains_operator_with_non_container(self):
        assert matches_filter(123, {"contains": "a"}) is False

    def test_gt_operator_matching(self):
        assert matches_filter(10, {"gt": 5}) is True

    def test_gt_operator_not_matching(self):
        assert matches_filter(5, {"gt": 5}) is False

    def test_gte_operator_matching(self):
        assert matches_filter(5, {"gte": 5}) is True

    def test_gte_operator_not_matching(self):
        assert matches_filter(4, {"gte": 5}) is False

    def test_lt_operator_matching(self):
        assert matches_filter(4, {"lt": 5}) is True

    def test_lt_operator_not_matching(self):
        assert matches_filter(5, {"lt": 5}) is False

    def test_lte_operator_matching(self):
        assert matches_filter(5, {"lte": 5}) is True

    def test_lte_operator_not_matching(self):
        assert matches_filter(6, {"lte": 5}) is False

    def test_exists_operator_with_existing_value(self):
        assert matches_filter("value", {"exists": True}) is True

    def test_exists_operator_with_none(self):
        assert matches_filter(None, {"exists": True}) is False

    def test_exists_operator_with_false(self):
        assert matches_filter(None, {"exists": False}) is True

    def test_nested_dict_filter(self):
        value = {"nested": {"key": "value"}}
        expected = {"nested": {"key": "value"}}
        assert matches_filter(value, expected) is True

    def test_nested_dict_filter_not_matching(self):
        value = {"nested": {"key": "other"}}
        expected = {"nested": {"key": "value"}}
        assert matches_filter(value, expected) is False

    def test_nested_dict_filter_missing_key(self):
        value = {"other": "value"}
        expected = {"nested": {"key": "value"}}
        assert matches_filter(value, expected) is False

    def test_list_filter(self):
        assert matches_filter("a", ["a", "b"]) is True

    def test_list_filter_not_matching(self):
        assert matches_filter("c", ["a", "b"]) is False

    def test_tuple_filter(self):
        assert matches_filter("a", ("a", "b")) is True

    def test_set_filter(self):
        assert matches_filter("a", {"a", "b"}) is True

    def test_direct_equality(self):
        assert matches_filter("value", "value") is True

    def test_direct_inequality(self):
        assert matches_filter("value", "other") is False

    def test_dict_expected_without_operators(self):
        # When expected is a dict but doesn't have operator keys
        # and value is not a dict, should return False
        assert matches_filter("string", {"key": "value"}) is False

    def test_multiple_operators(self):
        assert matches_filter(10, {"gte": 5, "lte": 15, "neq": 7}) is True

    def test_multiple_operators_failing(self):
        assert matches_filter(10, {"gte": 5, "lte": 8}) is False


# ---------------------------------------------------------------------------
# _escape_filter_value
# ---------------------------------------------------------------------------


class TestEscapeFilterValue:
    def test_escape_single_quote(self):
        assert escape_filter_value("user'1") == "user''1"

    def test_no_quotes(self):
        assert escape_filter_value("user1") == "user1"

    def test_multiple_quotes(self):
        assert escape_filter_value("it's") == "it''s"


# ---------------------------------------------------------------------------
# _build_filters edge cases
# ---------------------------------------------------------------------------


class TestBuildFilters:
    def test_build_filters_with_memory_type(self):
        filters = build_filters(
            user_id="user-1",
            session_id="",
            conversation_id="",
            memory_type="semantic",
            active_only=True,
        )
        assert "memory_type = 'semantic'" in filters
        assert "is_active = true" in filters

    def test_build_filters_without_active_only(self):
        filters = build_filters(
            user_id="user-1",
            session_id="",
            conversation_id="",
            active_only=False,
        )
        assert "is_active = true" not in filters

    def test_build_filters_with_session_id(self):
        filters = build_filters(
            user_id="user-1",
            session_id="session-1",
            conversation_id="",
        )
        assert "session_id = 'session-1'" in filters

    def test_build_filters_with_conversation_id(self):
        filters = build_filters(
            user_id="user-1",
            session_id="",
            conversation_id="conv-1",
        )
        assert "conversation_id = 'conv-1'" in filters

    def test_build_filters_with_quotes(self):
        filters = build_filters(
            user_id="user'1",
            session_id="",
            conversation_id="",
        )
        assert "user_id = 'user''1'" in filters
