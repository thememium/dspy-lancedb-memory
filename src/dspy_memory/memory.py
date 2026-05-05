"""Memory creation factory — a method-based API mirroring ``dspy_guardrails.guardrail``.

Usage
-----
::

    from dspy_memory import memory

    # 1. Configure once
    memory.configure(
        lm=dspy.LM("openrouter/anthropic/claude-sonnet-4-20250514"),
        embedding_model="openrouter/openai/text-embedding-3-small",
        reranker_model="cohere/rerank-4-fast",
    )

    # 2. Create a store
    store = memory.Store()

    # 3. Use it
    store.create_memory(user_id="u1", content="hello")
    results = store.search_memories(user_id="u1", query="hello", use_reranker=True)
"""

from __future__ import annotations

from typing import Any

from lancedb.rerankers import Reranker

from dspy_memory.config import configure as _configure
from dspy_memory.config import (
    get_embedding_config,
    get_reranker_config,
    get_signature_config,
    get_store_config,
)
from dspy_memory.reranking import OpenRouterReranker
from dspy_memory.store import LanceDSPyMemoryStore

# ---------------------------------------------------------------------------
# Sentinel to distinguish "not passed" from "explicitly None"
# ---------------------------------------------------------------------------

_UNSET: Any = object()


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


def configure(
    *,
    model: str | None = None,
    lm=None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    uri: str | None = None,
    table_name: str | None = None,
    signature=None,
    reranker_model: str | None = None,
    reranker_api_key: str | None = None,
):
    """Configure DSPy Memory globally.

    Call **once** at application startup.  Any parameter left as ``None``
    keeps its previous value (or the built-in default if never called).

    Parameters
    ----------
    model :
        Model string (e.g. ``"openrouter/anthropic/claude-sonnet-4-20250514"``).
    lm :
        Pre-built ``dspy.LM``.  Takes precedence over *model*.
    embedding_model :
        Embedding model (e.g. ``"openrouter/openai/text-embedding-3-small"``).
    embedding_dim :
        Output dimension of *embedding_model* (must match).
    uri :
        LanceDB URI (directory path).  Fallback for ``Store()``.
    table_name :
        LanceDB table name.  Fallback for ``Store()``.
    signature :
        A DSPy ``Signature`` subclass for memory extraction.  The signature
        must have ``memories: list[M]`` as its output field where ``M`` has
        ``.content`` and ``.type`` string attributes.
    reranker_model :
        Reranker model identifier
        (e.g. ``"cohere/rerank-4-fast"``).  ``None`` disables reranking.
    reranker_api_key :
        API key for the reranker.  Falls back to ``OPENROUTER_API_KEY`` env var.

    Example
    -------
    ::

        memory.configure(
            model="openrouter/anthropic/claude-sonnet-4-20250514",
            embedding_model="openrouter/openai/text-embedding-3-small",
            reranker_model="cohere/rerank-4-fast",
            uri=".my_memories",
            table_name="user_data",
        )
    """
    _configure(
        model=model,
        lm=lm,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        uri=uri,
        table_name=table_name,
        signature=signature,
        reranker_model=reranker_model,
        reranker_api_key=reranker_api_key,
    )


# ---------------------------------------------------------------------------
# Store() factory
# ---------------------------------------------------------------------------


def Store(
    uri: str | None | Any = _UNSET,
    table_name: str | None | Any = _UNSET,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    signature=None,
    reranker: Reranker | None | Any = _UNSET,
    rerank_limit_multiplier: int = 10,
) -> LanceDSPyMemoryStore:
    """Create a memory store, picking up defaults from :func:`configure`.

    Parameters
    ----------
    uri :
        LanceDB URI (directory path on disk).  Falls back to the value set
        in :func:`configure`, then to ``".lancedb"``.
    table_name :
        Name of the LanceDB table.  Falls back to :func:`configure`, then
        to ``"memories"``.
    embedding_model :
        Override the embedding model.  Falls back to the value set in
        :func:`configure` when ``None``.
    embedding_dim :
        Override the embedding dimension.  Falls back to :func:`configure` value.
    signature :
        Custom DSPy ``Signature`` subclass for extraction.  Falls back to
        :func:`configure`, then to the built-in ``ExtractMemory``.
    reranker :
        A LanceDB ``Reranker`` instance.

        * **Not passed** — reads *reranker_model* from :func:`configure` and,
          if set, auto-creates an ``OpenRouterReranker``.
        * ``None`` — explicitly disable reranking.
        * ``Reranker`` instance — use as-is.
    rerank_limit_multiplier :
        How many extra rows to fetch from vector search (before reranking
        trims back to the requested ``limit``).

    Returns
    -------
    LanceDSPyMemoryStore
    """
    if uri is _UNSET or table_name is _UNSET:
        cfg_uri, cfg_table = get_store_config()
        if uri is _UNSET:
            uri = cfg_uri
        if table_name is _UNSET:
            table_name = cfg_table
    assert isinstance(uri, str) and isinstance(table_name, str)

    if embedding_model is None or embedding_dim is None:
        cfg_embed, cfg_dim = get_embedding_config()
        if embedding_model is None:
            embedding_model = cfg_embed
        if embedding_dim is None:
            embedding_dim = cfg_dim

    if signature is None:
        signature = get_signature_config()

    if reranker is _UNSET:
        reranker_model, reranker_api_key = get_reranker_config()
        if reranker_model is not None:
            reranker = OpenRouterReranker(
                model=reranker_model,
                column="content",
                api_key=reranker_api_key,
            )
        else:
            reranker = None

    return LanceDSPyMemoryStore(
        uri=uri,
        table_name=table_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        signature=signature,
        reranker=reranker,
        rerank_limit_multiplier=rerank_limit_multiplier,
    )
