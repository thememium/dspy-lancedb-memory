from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MemoryItem(BaseModel):
    """One extracted memory returned by LLM."""

    content: str
    """The memory text — concise, self-contained fact, preference, or event."""

    type: str
    """
    Memory category.

    Must be one of: preference, semantic, episodic, procedural, summary, artifact.
    """

    metadata: dict[str, Any] = {}
    """Optional structured data attached to this extracted memory."""


class Memory(BaseModel):
    """A stored memory record as returned by the store.

    Use this in DSPy signatures via the ``Memories`` type alias::

        from dspy_lancedb_memory import Memories

        class MySignature(dspy.Signature):
            memories: Memories = dspy.InputField(
                description="Relevant memories from the store"
            )
    """

    id: str
    """Unique identifier for the memory row."""

    user_id: str
    """The user this memory belongs to."""

    session_id: str
    """Session or thread identifier (may be empty)."""

    conversation_id: str
    """Conversation identifier (may be empty)."""

    memory_type: str
    """Category string — e.g. ``"preference"``, ``"semantic"``, or a custom type."""

    content: str
    """The memory text."""

    relevance_score: float | None = None
    """Relevance score from search (cosine similarity or reranker score, 0–1). ``None`` when the memory was not retrieved via search."""

    metadata: dict[str, Any]
    """Arbitrary structured data attached at write time."""

    scope: dict[str, Any] = {}
    """Custom ownership/query dimensions."""

    replaces_id: str | None = None
    """ID of the memory this record replaces (append-only history chain). ``None`` for original memories."""

    is_active: bool = True
    """Whether this memory is the current active version. ``False`` for superseded records."""

    created_at: str
    """ISO-8601 timestamp of creation."""

    updated_at: str
    """ISO-8601 timestamp of last update."""


Memories = list[Memory]
"""Type alias for ``list[Memory]``.

Use in DSPy signatures::

    class MySignature(dspy.Signature):
        memories: Memories = dspy.InputField(
            description="Relevant memories from the store"
        )
"""
