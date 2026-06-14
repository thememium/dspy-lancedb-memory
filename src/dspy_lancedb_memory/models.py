from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class MemoryType(Enum):
    """Taxonomy of memory kinds extracted from conversation."""

    PREFERENCE = "preference"
    """User tastes, likes/dislikes, preferred formats, tone, tools."""

    SEMANTIC = "semantic"
    """Facts, biographical data, stable knowledge about the user."""

    EPISODIC = "episodic"
    """Events, tasks, decisions, or outcomes from a specific interaction."""

    PROCEDURAL = "procedural"
    """Learned rules, workflows, steps, patterns, or how-to knowledge."""

    SUMMARY = "summary"
    """Compressed conversation or task summaries capturing the gist."""

    ARTIFACT = "artifact"
    """Links, paths, IDs, or references to files, PRs, docs, outputs."""

    @classmethod
    def _missing_(cls, value: object):
        """Allow lookup by value string, e.g. MemoryType('semantic')."""
        for member in cls:
            if member.value == value:
                return member
        return None


class MemoryItem(BaseModel):
    """One extracted memory returned by the LLM."""

    content: str
    """The memory text — a concise, self-contained fact, preference, or event."""

    type: str
    """
    Memory category.

    Must be one of: preference, semantic, episodic, procedural, summary, artifact.
    """


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

    replaces_id: str | None = None
    """ID of the memory this record replaces (append-only history chain). ``None`` for original memories."""

    is_active: bool = True
    """Whether this memory is the current active version. ``False`` for superseded records."""

    created_at: str
    """ISO-8601 timestamp of creation."""

    updated_at: str
    """ISO-8601 timestamp of last update."""


class MemoryOperation(BaseModel):
    action: str
    content: str = ""
    search_query: str = ""
    memory_type: str = ""


class MemoryOperations(BaseModel):
    operations: list[MemoryOperation]


class ReconciledMemory(BaseModel):
    """Decision returned by the memory reconciler when comparing a new
    extracted memory against existing stored memories."""

    action: str
    memory_id: str = ""
    final_content: str = ""
    final_type: str = ""


Memories = list[Memory]
"""Type alias for ``list[Memory]``.

Use in DSPy signatures::

    class MySignature(dspy.Signature):
        memories: Memories = dspy.InputField(
            description="Relevant memories from the store"
        )
"""


def memory_type_from_string(value: MemoryType | str | None) -> MemoryType | str:
    """Coerce a raw value into a ``MemoryType`` enum member or pass through custom types."""
    if isinstance(value, MemoryType):
        return value
    if not value:
        return MemoryType.SEMANTIC
    for member in MemoryType:
        if member.value == value.lower():
            return member
    return value  # pass through unrecognised / custom types as-is
