"""Global configuration for DSPy Memory — extraction LM, embedding model, and reranker."""

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
# No default reranker — only enabled when explicitly configured.
DEFAULT_RERANKER_MODEL: str | None = None

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_lm: dspy.LM | None = None
_embedding_model: str = DEFAULT_EMBEDDING_MODEL
_embedding_dim: int = DEFAULT_EMBEDDING_DIM
_uri: str = DEFAULT_URI
_table_name: str = DEFAULT_TABLE_NAME
_reranker_model: str | None = DEFAULT_RERANKER_MODEL
_reranker_api_key: str | None = None
# Custom extraction signature class.  None → use built-in ExtractMemory.
_signature = None

# Whether configure() has been explicitly called at least once.
_configured: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure(
    *,
    model: str | None = None,
    lm: dspy.LM | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    uri: str | None = None,
    table_name: str | None = None,
    signature=None,
    reranker_model: str | None = None,
    reranker_api_key: str | None = None,
) -> dspy.LM | None:
    """Configure DSPy Memory globally.

    Call once at startup to set the LM used for memory extraction, the
    embedding model, LanceDB defaults, and (optionally) the reranker.
    Any parameter left as ``None`` keeps the previous value (or default
    if never called).

    Parameters
    ----------
    model :
        Model string passed to ``dspy.LM()``, e.g.
        ``"openrouter/anthropic/claude-sonnet-4-20250514"``.
    lm :
        Pre-constructed ``dspy.LM`` instance.  Takes precedence over *model*.
    embedding_model :
        Model for text embeddings, e.g.
        ``"openrouter/openai/text-embedding-3-small"``.
    embedding_dim :
        Output dimension of *embedding_model*.
    uri :
        LanceDB URI (directory path).  Used as default by ``memory.Store()``.
    table_name :
        LanceDB table name.  Used as default by ``memory.Store()``.
    signature :
        A DSPy ``Signature`` subclass to use for memory extraction instead of
        the built-in ``ExtractMemory``.  The signature must have an output
        field ``memories: list[M]`` where ``M`` has ``.content`` and ``.type``
        string attributes.
    reranker_model :
        Reranker model identifier, e.g. ``"cohere/rerank-4-fast"``.
        Pass ``None`` to disable reranking.
    reranker_api_key :
        API key for the reranker endpoint. Falls back to
        ``OPENROUTER_API_KEY`` env var if not set.

    Returns
    -------
    The configured ``dspy.LM`` for extraction, or ``None`` if neither
    *model* nor *lm* was provided.
    """
    global _lm, _embedding_model, _embedding_dim
    global _uri, _table_name
    global _reranker_model, _reranker_api_key, _signature, _configured

    if lm is not None:
        _lm = lm
        dspy.configure(lm=lm)
    elif model is not None:
        _lm = dspy.LM(model=model)
        dspy.configure(lm=_lm)

    if embedding_model is not None:
        _embedding_model = embedding_model
    if embedding_dim is not None:
        _embedding_dim = embedding_dim
    if uri is not None:
        _uri = uri
    if table_name is not None:
        _table_name = table_name
    if signature is not None:
        _signature = signature
    if reranker_model is not None:
        _reranker_model = reranker_model
    if reranker_api_key is not None:
        _reranker_api_key = reranker_api_key

    _configured = True
    return _lm


# ---------------------------------------------------------------------------
# Getters (used internally by the SDK)
# ---------------------------------------------------------------------------


def get_configured_lm() -> dspy.LM:
    """Return the configured LM or raise."""
    if _lm is None:
        raise RuntimeError(
            "No language model configured. Call memory.configure() first."
        )
    return _lm


def get_configured() -> bool:
    """Whether ``configure()`` has been called at least once."""
    return _configured


def get_embedding_config() -> tuple[str, int]:
    """Return ``(embedding_model, embedding_dim)``."""
    return _embedding_model, _embedding_dim


def get_store_config() -> tuple[str, str]:
    """Return ``(uri, table_name)``."""
    return _uri, _table_name


def get_reranker_config() -> tuple[str | None, str | None]:
    """Return ``(reranker_model, reranker_api_key)`` — both may be ``None``."""
    return _reranker_model, _reranker_api_key


def get_signature_config():
    """Return the configured signature class, or ``None`` (use built-in)."""
    return _signature
