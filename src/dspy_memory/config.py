"""Global configuration for DSPy Memory — extraction LM, embedding LM, and reranker."""

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
DEFAULT_NAMESPACE: list[str] = []

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_lm: dspy.LM | None = None
_embedding_lm: dspy.LM | None = None
_embedding_dim: int = DEFAULT_EMBEDDING_DIM
_uri: str = DEFAULT_URI
_table_name: str = DEFAULT_TABLE_NAME
_namespace: list[str] = DEFAULT_NAMESPACE.copy()
_reranker_lm: dspy.LM | str | None = None
_signature = None

_configured: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure(
    *,
    model: str | None = None,
    extraction_lm: dspy.LM | None = None,
    embedding_lm: dspy.LM | None = None,
    embedding_dim: int | None = None,
    uri: str | None = None,
    table_name: str | None = None,
    namespace: list[str] | None = None,
    signature=None,
    reranker_lm: dspy.LM | str | None = None,
) -> dspy.LM | None:
    """Configure DSPy Memory globally.

    Call once at startup.  Any parameter left as ``None`` keeps the
    previous value (or default if never called).

    Parameters
    ----------
    model :
        Model string (alternative to *extraction_lm* — creates ``dspy.LM(model)``).
    extraction_lm :
        ``dspy.LM`` for memory extraction.
    embedding_lm :
        ``dspy.LM`` for generating text embeddings.
    embedding_dim :
        Output dimension of the embedding model (must match the model).
    uri :
        LanceDB URI (directory path).  Default for ``memory.Store()``.
    table_name :
        LanceDB table name.  Default for ``memory.Store()``.
    namespace :
        LanceDB namespace path as a list of components, e.g.
        ``["prod", "search"]``.  The empty list ``[]`` means root
        namespace.  Namespace components must contain only letters,
        numbers, underscores, hyphens, and periods.
    signature :
        A DSPy ``Signature`` subclass for memory extraction instead of the
        built-in ``ExtractMemory``.
    reranker_lm :
        A ``dspy.LM`` or model string identifying the reranker model, e.g.
        ``dspy.LM("openrouter/cohere/rerank-4-fast")`` or
        ``"cohere/rerank-english-v3.0"``.  ``None`` disables reranking.

    Returns
    -------
    The configured ``dspy.LM`` for extraction, or ``None`` if not provided.
    """
    global _lm, _embedding_lm, _embedding_dim
    global _uri, _table_name, _namespace, _reranker_lm, _signature, _configured

    if extraction_lm is not None:
        _lm = extraction_lm
        dspy.configure(lm=extraction_lm)
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
    if namespace is not None:
        _namespace = list(namespace)
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


def get_store_config() -> tuple[str, str, list[str]]:
    return _uri, _table_name, list(_namespace)


def get_namespace_config() -> list[str]:
    return list(_namespace)


def get_reranker_lm_config() -> dspy.LM | str | None:
    return _reranker_lm


def get_signature_config():
    return _signature
