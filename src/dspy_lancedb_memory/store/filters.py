"""Python-side filtering for JSON metadata and scope dimensions."""

from __future__ import annotations

from typing import Any

from ..models import ScopeLike
from .serialization import scope_dict, split_metadata_scope

_FILTER_OPERATORS = frozenset(
    {"eq", "neq", "in", "contains", "gt", "gte", "lt", "lte", "exists"}
)


def escape_filter_value(value: str) -> str:
    """Escape a value for LanceDB's single-quoted SQL filter literals."""
    return value.replace("'", "''")


def matches_filter(value: Any, expected: Any) -> bool:
    """Match *value* against a scalar, collection, or operator expression."""
    if isinstance(expected, dict):
        if any(key in _FILTER_OPERATORS for key in expected):
            for operator, operand in expected.items():
                if operator == "eq" and value != operand:
                    return False
                if operator == "neq" and value == operand:
                    return False
                if operator == "in" and value not in operand:
                    return False
                if operator == "contains":
                    if isinstance(value, (list, tuple, set)):
                        if operand not in value:
                            return False
                    elif isinstance(value, str):
                        if str(operand) not in value:
                            return False
                    else:
                        return False
                if operator == "gt" and not (value > operand):
                    return False
                if operator == "gte" and not (value >= operand):
                    return False
                if operator == "lt" and not (value < operand):
                    return False
                if operator == "lte" and not (value <= operand):
                    return False
                if operator == "exists" and ((value is not None) is not bool(operand)):
                    return False
            return True

        if not isinstance(value, dict):
            return False
        return all(
            key in value and matches_filter(value[key], nested)
            for key, nested in expected.items()
        )
    if isinstance(expected, (list, tuple, set)):
        return value in expected
    return value == expected


def row_matches_json_filters(
    row: dict[str, Any],
    *,
    scope: ScopeLike | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> bool:
    """Return whether a stored row matches scope and metadata filters."""
    metadata, row_scope = split_metadata_scope(row.get("metadata"))
    scope_filter = scope_dict(scope)
    if scope_filter and not matches_filter(row_scope, scope_filter):
        return False
    return not metadata_filter or matches_filter(metadata, metadata_filter)


def filter_rows_by_json_fields(
    rows: list[dict[str, Any]],
    *,
    scope: ScopeLike | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter rows that LanceDB cannot express as native where clauses."""
    scope_filter = scope_dict(scope)
    if not scope_filter and not metadata_filter:
        return rows
    return [
        row
        for row in rows
        if row_matches_json_filters(
            row, scope=scope_filter, metadata_filter=metadata_filter
        )
    ]


def build_filters(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    memory_type: str | None = None,
    active_only: bool = True,
) -> list[str]:
    """Build the native LanceDB filter clauses shared by all queries."""
    filters = [f"user_id = '{escape_filter_value(user_id)}'"]

    if session_id:
        filters.append(f"session_id = '{escape_filter_value(session_id)}'")

    if conversation_id:
        filters.append(f"conversation_id = '{escape_filter_value(conversation_id)}'")

    if memory_type:
        filters.append(f"memory_type = '{escape_filter_value(memory_type)}'")

    if active_only:
        filters.append("is_active = true")

    return filters
