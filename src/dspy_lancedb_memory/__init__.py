import os

# LanceDB's LSM scanner emits a per-query deprecation warning that the Python
# API cannot disable per query.  Silence that target unless the app configured
# LanceDB logging explicitly.  This must run before ``lancedb`` is imported.
os.environ.setdefault("LANCEDB_LOG", "lance::dataset::scanner=error,lancedb=warn")

from . import memory
from .extraction import (
    ExtractMemory,
    ExtractMemoryOperations,
    MemoryExtractor,
    MemoryOperationExtractor,
    MemoryReconciler,
)
from .models import (
    Memories,
    Memory,
    MemoryItem,
    MemoryOperation,
    MemoryOperations,
    MemoryType,
    PendingReconciliation,
    ReconciledMemory,
    Scope,
    ScopeLike,
)
from .reranking import LiteLLMReranker
from .store import BoundMemoryStore, LanceDSPyMemoryStore

__all__ = [
    "BoundMemoryStore",
    "ExtractMemory",
    "ExtractMemoryOperations",
    "LanceDSPyMemoryStore",
    "LiteLLMReranker",
    "Memories",
    "Memory",
    "MemoryExtractor",
    "MemoryItem",
    "MemoryOperation",
    "MemoryOperationExtractor",
    "MemoryOperations",
    "MemoryReconciler",
    "MemoryType",
    "PendingReconciliation",
    "ReconciledMemory",
    "Scope",
    "ScopeLike",
    "memory",
]
