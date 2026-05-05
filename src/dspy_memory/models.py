from enum import Enum

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
