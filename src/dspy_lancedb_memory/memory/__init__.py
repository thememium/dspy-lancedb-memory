"""Memory creation factory — configure once, then build stores.

Usage
-----
::

    from dspy_lancedb_memory import memory

    memory.configure(
        extraction_lm=dspy.LM("openrouter/anthropic/claude-sonnet-4-20250514"),
        embedding_lm=dspy.LM("openrouter/openai/text-embedding-3-small"),
        reranker_lm=dspy.LM("openrouter/cohere/rerank-4-fast"),
    )

    store = memory.Store()
"""

from ..config.state import configure
from .store_factory import Store

__all__ = ["Store", "configure"]
