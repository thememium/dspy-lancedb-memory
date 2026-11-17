"""``Store()`` factory resolving configuration defaults."""

from __future__ import annotations

from typing import Any

import dspy
from lancedb.rerankers import Reranker

from ..config.defaults import DEFAULT_EMBEDDING_MODEL
from ..config.state import (
    get_configured_lm,
    get_embedding_config,
    get_reranker_lm_config,
    get_signature_config,
    get_store_config,
)
from ..store import LanceDSPyMemoryStore
from .reranker import build_reranker

_UNSET: Any = object()


def Store(
    uri: str | None | Any = _UNSET,
    table_name: str | None | Any = _UNSET,
    embedding_lm=None,
    embedding_dim: int | None = None,
    signature=None,
    reranker_lm: dspy.LM | str | None = None,
    reranker: Reranker | None | Any = _UNSET,
    rerank_limit_multiplier: int = 10,
    rte_durable_write: bool = False,
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
    rte_durable_write :
        When ``True``, every write waits for WAL persistence (~100 ms per write
        on macOS, a few ms on Linux).  Default ``False`` batches WAL flushes
        (100 ms / 10 MB), so writes are sub-millisecond and a crash can lose
        only that tail.  Applies when the table first enables RTE.

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
        reranker = build_reranker(reranker_lm)

    return LanceDSPyMemoryStore(
        uri=uri,
        table_name=table_name,
        extraction_lm=get_configured_lm(),
        embedding_lm=embedding_lm,
        embedding_dim=embedding_dim,
        signature=signature,
        reranker=reranker,
        rerank_limit_multiplier=rerank_limit_multiplier,
        rte_durable_write=rte_durable_write,
    )
