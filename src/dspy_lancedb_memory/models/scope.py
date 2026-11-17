from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Scope(BaseModel):
    """Typed helper for arbitrary custom memory ownership/query dimensions."""

    model_config = ConfigDict(extra="allow")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return scope fields as a plain dict."""
        return self.model_dump(mode="json")


ScopeLike = Scope | BaseModel | dict[str, Any]
"""Accepted input type for custom memory scope values."""
