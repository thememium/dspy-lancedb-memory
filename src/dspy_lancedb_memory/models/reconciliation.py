from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .records import Memory


class ReconciledMemory(BaseModel):
    """Decision returned by the memory reconciler when comparing a new
    extracted memory against existing stored memories."""

    action: str
    memory_id: str = ""
    final_content: str = ""
    final_type: str = ""


@dataclass
class PendingReconciliation:
    """Holds reconciliation decision until writes are applied.

    Used by the two-phase parallel reconciliation pipeline:
    Phase 1 (parallel) populates this; Phase 2 (sequential) applies writes.
    """

    content: str
    inferred_type: str
    decision: ReconciledMemory
    user_id: str
    session_id: str
    conversation_id: str
    scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    existing_row: Memory | None = None
