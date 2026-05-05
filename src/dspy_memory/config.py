"""Global configuration for DSPy Memory — extraction LM, embedding LM, and reranker LM."""

from __future__ import annotations

import dspy

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LM_MODEL = "openrouter/openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_URI = ".lancedb"
DEFAULT_TABLE_NAME = "memories"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_lm: dspy.LM | None = None
_embedding_lm: dspy.LM | None = None
_embedding_dim: int = DEFAULT_EMBEDDING_DIM
_uri: str = DEFAULT_URI
_table_name: str = DEFAULT_TABLE_NAME
_reranker_lm: dspy.LM | None = None
_signature = None

_configured: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure(
    *,
    model: str | None = None,
    lm: dspy.LM | None = None,
    embedding_lm: dspy.LM | None = None,
    embedding_dim: int | None = None,
    uri: str | None = None,
    table_name: str | None = None,
    signature=None,
    reranker_lm: dspy.LM | None = None,
) -> dspy.LM | None:
    """Configure DSPy Memory globally.

    Call once at startup.  Every parameter is a ``dspy.LM`` instance.
    Any parameter left as ``None`` keeps the previous value (or default
    if never called).

    Parameters
    ----------
    model :
        Model string (alternative to *lm* — creates ``dspy.LM(model)``).
    lm :
        ``dspy.LM`` for memory extraction.
    embedding_lm :
        ``dspy.LM`` for generating text embeddings.
    embedding_dim :
        Output dimension of the embedding model (must match the model).
    uri :
        LanceDB URI (directory path).  Default for ``memory.Store()``.
    table_name :
        LanceDB table name.  Default for ``memory.Store()``.
    signature :
        A DSPy ``Signature`` subclass for memory extraction instead of the
        built-in ``ExtractMemory``.
    reranker_lm :
        ``dspy.LM`` identifying the reranker model (e.g. ``dspy.LM("openrouter/cohere/rerank-4-fast")``).
        ``None`` disables reranking.

    Returns
    -------
    The configured ``dspy.LM`` for extraction, or ``None`` if not provided.
    """
    global _lm, _embedding_lm, _embedding_dim
    global _uri, _table_name
    global _reranker_lm, _signature, _configured

    if lm is not None:
        _lm = lm
        dspy.configure(lm=lm)
    elif model is not None:
        _lm = dspy.LM(model=model)
        dspy.configure(lm=_lm)

    if embedding_lm is not None:
        _embedding_lm = embedding_lm
    if embedding_dim is not None:
        _embedding_dim = embedding_dim
    if uri is not None:
        _uri = uri
    if table_name is not None:
        _table_name = table_name
    if signature is not None:
        _signature = signature
    if reranker_lm is not None:
        _reranker_lm = reranker_lm

    _configured = True
    return _lm


# ---------------------------------------------------------------------------
# Getters (used internally by the SDK)
# ---------------------------------------------------------------------------


def get_configured_lm() -> dspy.LM:
    if _lm is None:
        raise RuntimeError(
            "No language model configured. Call memory.configure() first."
        )
    return _lm


def get_configured() -> bool:
    return _configured


def get_embedding_config() -> tuple[dspy.LM | None, int]:
    return _embedding_lm, _embedding_dim


def get_store_config() -> tuple[str, str]:
    return _uri, _table_name


def get_reranker_lm_config() -> dspy.LM | None:
    return _reranker_lm


def get_signature_config():
    return _signature
