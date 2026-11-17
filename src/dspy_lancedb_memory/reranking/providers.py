"""Built-in rerank providers and credential lookup."""

from __future__ import annotations

import os
import re

# Built-in rerank endpoints and the environment variables that carry their keys.
# Providers without an entry require an explicit ``api_base``.
RERANK_PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cohere": ("https://api.cohere.com/v2", ("COHERE_API_KEY", "CO_API_KEY")),
    "jina": ("https://api.jina.ai/v1", ("JINA_API_KEY",)),
    "together_ai": (
        "https://api.together.xyz/v1",
        ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY"),
    ),
    "voyage": ("https://api.voyageai.com/v1", ("VOYAGE_API_KEY",)),
    "fireworks_ai": ("https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",)),
}

# OpenRouter names Cohere generations ``cohere/rerank-4-fast``; Cohere's direct
# API names them ``cohere/rerank-v4.0-fast``.  The unsigned generation suffix
# therefore identifies an OpenRouter model without requiring a prefix.
_OPENROUTER_MODEL = re.compile(r"^(?:cohere/)?rerank-\d")


def provider_of(model: str) -> str:
    """Return the routing prefix of *model* (``"cohere/x"`` → ``"cohere"``)."""
    return model.split("/", 1)[0] if "/" in model else model


def model_name_of(model: str) -> str:
    """Return the model id without the routing prefix."""
    return model.split("/", 1)[1] if "/" in model else model


def is_openrouter_model(model: str) -> bool:
    """OpenRouter is selected by prefix or by Cohere's OpenRouter naming."""
    return model.startswith("openrouter/") or bool(_OPENROUTER_MODEL.match(model))


def provider_api_key(provider: str, explicit_key: str | None = None) -> str | None:
    """Resolve an API key from an explicit value or provider environment vars."""
    if explicit_key:
        return explicit_key
    spec = RERANK_PROVIDERS.get(provider)
    if spec is None:
        return None
    for env_var in spec[1]:
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def openrouter_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")
