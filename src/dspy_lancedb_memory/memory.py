"""Memory creation factory — a method-based API mirroring ``dspy_guardrails.guardrail``.

Usage
-----
::

    from dspy_lancedb_memory import memory
    import dspy

    # 1. Configure once
    memory.configure(
        extraction_lm=dspy.LM("openrouter/anthropic/claude-sonnet-4-20250514"),
        embedding_lm=dspy.LM("openrouter/openai/text-embedding-3-small"),
        reranker_model=dspy.LM("openrouter/cohere/rerank-4-fast"),
    )

    # 2. Create a store
    store = memory.Store()

    # 3. Use it
    store.create_memory(user_id="u1", content="hello")
    results = store.search_memories(user_id="u1", query="hello", use_reranker=True)
"""

from __future__ import annotations

from typing import Any

import dspy
from lancedb.rerankers import Reranker

from dspy_lancedb_memory.config import DEFAULT_EMBEDDING_MODEL
from dspy_lancedb_memory.config import configure as _configure
from dspy_lancedb_memory.config import (
    get_configured_lm,
    get_embedding_config,
    get_reranker_lm_config,
    get_signature_config,
    get_store_config,
)
from dspy_lancedb_memory.reranking import LiteLLMReranker
from dspy_lancedb_memory.store import LanceDSPyMemoryStore

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
    extraction_lm=None,
    embedding_lm: dspy.LM | None = None,
    embedding_dim: int | None = None,
    uri: str | None = None,
    table_name: str | None = None,
    signature=None,
    reranker_lm: dspy.LM | str | None = None,
):
    """Configure DSPy Memory globally.

    Call **once** at application startup.  Any parameter left as ``None``
    keeps its previous value (or the built-in default if never called).

    Parameters
    ----------
    model :
        Model string (alternative to *extraction_lm* — creates ``dspy.LM(model)``).
    extraction_lm :
        ``dspy.LM`` for memory extraction.
    embedding_lm :
        ``dspy.LM`` for text embeddings.
    embedding_dim :
        Output dimension of the embedding model (must match).
    uri :
        LanceDB URI (directory path).  Fallback for ``Store()``.
    table_name :
        LanceDB table name.  Fallback for ``Store()``.
    signature :
        A DSPy ``Signature`` subclass for memory extraction.
    reranker_lm :
        A ``dspy.LM`` or model string for the reranker
        (e.g. ``dspy.LM("openrouter/cohere/rerank-4-fast")``).

    Example
    -------
    ::

        import dspy

        memory.configure(
            extraction_lm=dspy.LM("openrouter/anthropic/claude-sonnet-4-20250514"),
            embedding_lm=dspy.LM("openrouter/openai/text-embedding-3-small"),
            reranker_lm=dspy.LM("openrouter/cohere/rerank-4-fast"),
            uri=".my_memories",
            table_name="user_data",
        )
    """
    _configure(
        model=model,
        extraction_lm=extraction_lm,
        embedding_lm=embedding_lm,
        embedding_dim=embedding_dim,
        uri=uri,
        table_name=table_name,
        signature=signature,
        reranker_lm=reranker_lm,
    )


# ---------------------------------------------------------------------------
# Store() factory
# ---------------------------------------------------------------------------


def Store(
    uri: str | None | Any = _UNSET,
    table_name: str | None | Any = _UNSET,
    embedding_lm=None,
    embedding_dim: int | None = None,
    signature=None,
    reranker_lm: dspy.LM | str | None = None,
    reranker: Reranker | None | Any = _UNSET,
    rerank_limit_multiplier: int = 10,
) -> LanceDSPyMemoryStore:
    """Create a memory store, picking up defaults from :func:`configure`.

    Parameters
    ----------
    uri :
        LanceDB URI.  Falls back to :func:`configure`, then ``".lancedb"``.
    table_name :
        LanceDB table name.  Falls back to :func:`configure`, then ``"memories"``.
    embedding_lm :
        ``dspy.LM`` for embeddings.  Falls back to :func:`configure`, then
        ``dspy.LM("openrouter/openai/text-embedding-3-small")``.
    embedding_dim :
        Embedding dimension.  Falls back to :func:`configure`, then ``1536``.
    signature :
        Custom DSPy ``Signature`` for extraction.  Falls back to
        :func:`configure`, then the built-in ``ExtractMemory``.
    reranker_lm :
        A ``dspy.LM`` or model string for ``LiteLLMReranker``.  Falls back
        to :func:`configure`.  Ignored if *reranker* is passed.
    reranker :
        A LanceDB ``Reranker`` instance.  Takes precedence over
        *reranker_lm*.

        * **Not passed** — creates a ``LiteLLMReranker`` from *reranker_lm*.
        * ``None`` — explicitly disable reranking.
        * ``Reranker`` instance — use as-is.
    rerank_limit_multiplier :
        Extra candidates to fetch before reranking trims to ``limit``.

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

    if embedding_lm is None or embedding_dim is None:
        cfg_lm, cfg_dim = get_embedding_config()
        if embedding_lm is None:
            embedding_lm = cfg_lm or dspy.LM(DEFAULT_EMBEDDING_MODEL)
        if embedding_dim is None:
            embedding_dim = cfg_dim

    if signature is None:
        signature = get_signature_config()

    if reranker_lm is None:
        reranker_lm = get_reranker_lm_config()

    if reranker is _UNSET:
        if reranker_lm is not None:
            model_str = (
                reranker_lm.model if isinstance(reranker_lm, dspy.LM) else reranker_lm
            )
            reranker_kwargs: dict = {}
            if isinstance(reranker_lm, dspy.LM):
                lm_kwargs: dict = getattr(reranker_lm, "kwargs", {})
                if lm_kwargs.get("api_base"):
                    reranker_kwargs["api_base"] = lm_kwargs["api_base"]
                if lm_kwargs.get("api_key"):
                    reranker_kwargs["api_key"] = lm_kwargs["api_key"]
            reranker = LiteLLMReranker(
                model=model_str,
                column="content",
                **reranker_kwargs,
            )
        else:
            reranker = None

    return LanceDSPyMemoryStore(
        uri=uri,
        table_name=table_name,
        extraction_lm=get_configured_lm(),
        embedding_lm=embedding_lm,
        embedding_dim=embedding_dim,
        signature=signature,
        reranker=reranker,
        rerank_limit_multiplier=rerank_limit_multiplier,
    )
