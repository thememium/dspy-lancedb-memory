from __future__ import annotations

from pydantic import BaseModel


class MemoryOperation(BaseModel):
    action: str
    content: str = ""
    search_query: str = ""
    memory_type: str = ""


class MemoryOperations(BaseModel):
    operations: list[MemoryOperation]
