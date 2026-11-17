"""JSON encoding helpers for memory metadata and scope fields."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ..models import Scope, ScopeLike

SCOPE_METADATA_KEY = "_scope"


def json_dict(value: Any) -> dict[str, Any]:
    """Coerce JSON text or a dict into a plain dictionary."""
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def scope_dict(scope: ScopeLike | None) -> dict[str, Any]:
    """Coerce any accepted scope shape into a plain dictionary."""
    if scope is None:
        return {}
    if isinstance(scope, Scope):
        return scope.to_dict()
    if isinstance(scope, BaseModel):
        return scope.model_dump(mode="json")
    return dict(scope)


def split_metadata_scope(
    metadata: dict[str, Any] | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate stored metadata from the reserved scope entry."""
    clean_metadata = json_dict(metadata)
    scope = json_dict(clean_metadata.pop(SCOPE_METADATA_KEY, {}))
    return clean_metadata, scope


def pack_metadata(
    metadata: dict[str, Any] | None,
    scope: ScopeLike | None,
) -> str:
    """Encode metadata and optional scope as the stored JSON string."""
    packed = dict(metadata or {})
    if SCOPE_METADATA_KEY in packed:
        raise ValueError(
            f"metadata key {SCOPE_METADATA_KEY!r} is reserved; use scope= instead"
        )
    scope_fields = scope_dict(scope)
    if scope_fields:
        packed[SCOPE_METADATA_KEY] = scope_fields
    return json.dumps(packed)


def merge_metadata(
    base: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(extra or {})
    return merged
