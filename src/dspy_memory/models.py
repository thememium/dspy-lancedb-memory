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

        from dspy_memory import Memories

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

    metadata: dict[str, Any]
    """Arbitrary structured data attached at write time."""

    created_at: str
    """ISO-8601 timestamp of creation."""

    updated_at: str
    """ISO-8601 timestamp of last update."""


class ReconciledMemory(BaseModel):
    """Decision returned by the memory reconciler when comparing a new
    extracted memory against existing stored memories."""

    action: str
    """One of ``"keep"``, ``"update"``, or ``"create"``."""

    memory_id: str = ""
    """The existing memory ``id`` if *action* is ``"keep"`` or ``"update"``."""

    final_content: str = ""
    """
    The content that should be stored.

    * ``"keep"`` — same as the existing memory’s content.
    * ``"update"`` — a synthesized combination of old + new information.
    * ``"create"`` — the new memory’s content.
    """

    final_type: str = ""
    """Memory type string (same as ``new_memory_type`` for ``"create"``)."""


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
