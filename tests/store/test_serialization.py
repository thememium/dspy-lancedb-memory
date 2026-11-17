"""Additional tests for dspy_lancedb_memory.store — covering remaining uncovered paths."""

from __future__ import annotations

import json

import pytest

from dspy_lancedb_memory.models import Scope
from dspy_lancedb_memory.store.serialization import (
    json_dict,
    pack_metadata,
    scope_dict,
    split_metadata_scope,
)

# ---------------------------------------------------------------------------
# _json_dict edge cases
# ---------------------------------------------------------------------------


class TestJsonDict:
    def test_json_dict_with_dict_input(self):
        d = {"key": "value"}
        result = json_dict(d)
        assert result == d
        # Should be a copy, not the same object
        assert result is not d

    def test_json_dict_with_empty_dict(self):
        result = json_dict({})
        assert result == {}

    def test_json_dict_with_none(self):
        result = json_dict(None)
        assert result == {}

    def test_json_dict_with_empty_string(self):
        result = json_dict("")
        assert result == {}

    def test_json_dict_with_valid_json_string(self):
        result = json_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_dict_with_invalid_json_string(self):
        result = json_dict("not json")
        assert result == {}

    def test_json_dict_with_json_array(self):
        # JSON array should return empty dict (not a dict)
        result = json_dict("[1, 2, 3]")
        assert result == {}

    def test_json_dict_with_integer(self):
        result = json_dict(0)
        assert result == {}

    def test_json_dict_with_list(self):
        result = json_dict([])
        assert result == {}

    def test_json_dict_with_truthy_integer(self):
        """Test _json_dict with truthy non-dict, non-string value (line 214)."""
        result = json_dict(42)
        assert result == {}

    def test_json_dict_with_boolean_true(self):
        """Test _json_dict with boolean True."""
        result = json_dict(True)
        assert result == {}

    def test_json_dict_with_non_empty_list(self):
        """Test _json_dict with non-empty list (truthy but not dict/string)."""
        result = json_dict([1, 2, 3])
        assert result == {}


# ---------------------------------------------------------------------------
# _split_metadata_scope
# ---------------------------------------------------------------------------


class TestSplitMetadataScope:
    def test_split_with_scope_key(self):
        metadata = '{"_scope": {"tenant": "a"}, "source": "chat"}'
        clean, scope = split_metadata_scope(metadata)
        assert clean == {"source": "chat"}
        assert scope == {"tenant": "a"}

    def test_split_without_scope_key(self):
        metadata = '{"source": "chat"}'
        clean, scope = split_metadata_scope(metadata)
        assert clean == {"source": "chat"}
        assert scope == {}

    def test_split_with_dict_input(self):
        metadata = {"_scope": {"tenant": "a"}, "source": "chat"}
        clean, scope = split_metadata_scope(metadata)
        assert clean == {"source": "chat"}
        assert scope == {"tenant": "a"}

    def test_split_with_none(self):
        clean, scope = split_metadata_scope(None)
        assert clean == {}
        assert scope == {}


# ---------------------------------------------------------------------------
# _pack_metadata
# ---------------------------------------------------------------------------


class TestPackMetadata:
    def test_pack_with_metadata_and_scope(self):
        result = pack_metadata({"source": "chat"}, {"tenant": "a"})
        parsed = json.loads(result)
        assert parsed["source"] == "chat"
        assert parsed["_scope"] == {"tenant": "a"}

    def test_pack_with_empty_metadata(self):
        result = pack_metadata(None, None)
        assert json.loads(result) == {}

    def test_pack_with_reserved_key_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            pack_metadata({"_scope": {}}, None)


# ---------------------------------------------------------------------------
# _scope_dict edge cases
# ---------------------------------------------------------------------------


class TestScopeDict:
    def test_scope_dict_with_none(self):
        assert scope_dict(None) == {}

    def test_scope_dict_with_scope_model(self):
        scope = Scope(tenant="a", repo="b")
        result = scope_dict(scope)
        assert result == {"tenant": "a", "repo": "b"}

    def test_scope_dict_with_dict(self):
        result = scope_dict({"tenant": "a"})
        assert result == {"tenant": "a"}
