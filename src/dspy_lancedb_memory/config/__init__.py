"""Global configuration API for DSPy Memory."""

from .defaults import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LM_MODEL,
    DEFAULT_TABLE_NAME,
    DEFAULT_URI,
)
from .state import (
    configure,
    get_configured,
    get_configured_lm,
    get_embedding_config,
    get_reranker_lm_config,
    get_signature_config,
    get_store_config,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LM_MODEL",
    "DEFAULT_TABLE_NAME",
    "DEFAULT_URI",
    "configure",
    "get_configured",
    "get_configured_lm",
    "get_embedding_config",
    "get_reranker_lm_config",
    "get_signature_config",
    "get_store_config",
]
