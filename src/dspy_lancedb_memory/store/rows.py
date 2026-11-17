"""Translation between LanceDB rows and memory models."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from ..models import (
    Memories,
    Memory,
    MemoryType,
    ScopeLike,
    memory_type_from_string,
    memory_type_value,
)
from .serialization import pack_metadata, split_metadata_scope
from .similarity import distance_to_similarity


def build_memory_row(
    *,
    embed: Callable[[str], list[float]],
    user_id: str,
    content: str,
    session_id: str = "",
    conversation_id: str = "",
    memory_type: MemoryType | str,
    metadata: dict[str, Any] | None,
    scope: ScopeLike | None = None,
) -> dict[str, Any]:
    """Build a fresh, active memory row, embedding *content*."""
    now = datetime.now(UTC).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "memory_type": memory_type_value(memory_type),
        "content": content,
        "metadata": pack_metadata(metadata, scope),
        "created_at": now,
        "updated_at": now,
        "replaces_id": None,
        "is_active": True,
        "vector": embed(content),
    }


def to_memory(row: dict[str, Any]) -> Memory:
    """Convert a raw LanceDB row dict to a ``Memory`` instance."""
    clean = dict(row)
    clean.pop("vector", None)

    # Preserve relevance score: reranker sets _relevance_score directly;
    # otherwise derive cosine similarity from LanceDB's _distance.
    relevance_score = None
    if "_relevance_score" in clean:
        relevance_score = float(clean.pop("_relevance_score"))
    elif "_distance" in clean:
        distance = float(clean.pop("_distance"))
        relevance_score = distance_to_similarity(distance)

    metadata, scope = split_metadata_scope(clean.get("metadata"))
    clean["metadata"] = metadata
    clean["scope"] = scope

    memory = Memory.model_validate(clean)
    memory.relevance_score = relevance_score
    return memory


def without_vectors(rows: list[Any]) -> Memories:
    """Convert rows to memories, passing through existing ``Memory`` values."""
    result: list[Memory] = []
    for row in rows:
        if isinstance(row, Memory):
            result.append(row)
        else:
            result.append(to_memory(row))
    return result


def importance_from_metadata(metadata: Any) -> float:
    """Resolve a numeric importance in ``[0, 1]`` from stored metadata."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    raw = metadata.get("_importance", metadata.get("importance"))
    if raw is None:
        return 0.5
    if isinstance(raw, (int, float)):
        return max(0.0, min(float(raw), 1.0))
    return {
        "critical": 1.0,
        "high": 0.85,
        "medium": 0.6,
        "normal": 0.5,
        "low": 0.3,
        "trivial": 0.1,
    }.get(str(raw).strip().lower(), 0.5)


def carry_legacy_fields(
    new_row: dict[str, Any],
    old_row: dict[str, Any],
    *,
    keep_search_text: bool,
) -> None:
    """Keep access stats (and optionally enriched search text) on a new version."""
    if keep_search_text and old_row.get("search_text") is not None:
        new_row["search_text"] = old_row["search_text"]
    for name in ("last_accessed_at", "access_count"):
        if old_row.get(name) is not None:
            new_row[name] = old_row[name]


def type_default(field_type: pa.DataType) -> Any:
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return ""
    if pa.types.is_boolean(field_type):
        return False
    if pa.types.is_integer(field_type):
        return 0
    if pa.types.is_floating(field_type):
        return 0.0
    if pa.types.is_fixed_size_list(field_type):
        return [0.0] * field_type.list_size
    if pa.types.is_list(field_type):
        return []
    return None


def default_for_field(field: pa.Field, row: dict[str, Any]) -> Any:
    """Fill columns present in older table schemas but not in *row*."""
    if field.name == "search_text":
        return row.get("content")
    if field.name in {"last_accessed_at", "access_count"}:
        return None
    if field.name == "importance":
        return importance_from_metadata(row.get("metadata"))
    if field.nullable:
        return None
    return type_default(field.type)


def row_for_write(row: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    """Project *row* onto the live table schema, filling legacy columns."""
    return {
        field.name: (
            row[field.name] if field.name in row else default_for_field(field, row)
        )
        for field in schema
    }


def normalize_extracted_memories(
    extracted: list[Any],
) -> list[tuple[str, MemoryType | str, dict[str, Any]]]:
    """Normalize extractor output into ``(content, type, metadata)`` tuples."""
    normalized: list[tuple[str, MemoryType | str, dict[str, Any]]] = []
    seen_contents: set[str] = set()
    for item in extracted:
        item_metadata: dict[str, Any] = {}
        if isinstance(item, tuple):
            if len(item) == 2:
                content, inferred_type = item
            elif len(item) >= 3:
                content, inferred_type, item_metadata = item[:3]
            else:
                continue
        else:
            content = getattr(item, "content", "")
            inferred_type = getattr(item, "type", MemoryType.SEMANTIC)
            item_metadata = dict(getattr(item, "metadata", {}) or {})
        content_value = str(content).strip()
        if not content_value or content_value in seen_contents:
            continue
        seen_contents.add(content_value)
        normalized.append(
            (
                content_value,
                memory_type_from_string(inferred_type),
                dict(item_metadata or {}),
            )
        )
    return normalized
