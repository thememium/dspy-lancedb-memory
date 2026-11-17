"""Embedding client construction and store embedding helpers."""

from __future__ import annotations

import os
from typing import Any

import dspy

from ..config.defaults import DEFAULT_EMBEDDING_MODEL
from .table import get_vector_dim


def build_embedder(embedding_lm) -> dspy.Embedder:
    """Build a ``dspy.Embedder``, routing OpenRouter and custom endpoints."""
    if embedding_lm is None:
        embedding_lm = dspy.LM(DEFAULT_EMBEDDING_MODEL)
    lm_kwargs: dict = getattr(embedding_lm, "kwargs", {})
    embedder_kwargs: dict = {}
    if lm_kwargs.get("api_base"):
        embedder_kwargs["api_base"] = lm_kwargs["api_base"]
    if lm_kwargs.get("api_key"):
        embedder_kwargs["api_key"] = lm_kwargs["api_key"]
    # ``dspy.Embedder`` only accepts OpenAI/Azure hosted models, so route
    # OpenRouter and custom endpoints through the OpenAI-compatible client.
    embedder_model = embedding_lm.model
    if embedder_model.startswith("openrouter/"):
        # ``dspy.Embedder`` drops the first path segment before calling the
        # API, so prefix the full OpenRouter id (e.g.
        # ``openai/openai/text-embedding-3-small`` sends
        # ``openai/text-embedding-3-small``).
        embedder_model = "openai/" + embedder_model[len("openrouter/") :]
        embedder_kwargs.setdefault("api_base", "https://openrouter.ai/api/v1")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            embedder_kwargs.setdefault("api_key", openrouter_key)
        embedder_kwargs.setdefault("encoding_format", "float")
    elif "api_base" in embedder_kwargs and "/" in embedder_model:
        embedder_model = "openai/" + embedder_model.split("/", 1)[1]
        embedder_kwargs["encoding_format"] = "float"
    return dspy.Embedder(
        embedder_model,
        caching=True,
        **embedder_kwargs,
    )


class EmbeddingMixin:
    """Embedding and vector-dimension helpers for the memory store."""

    db: Any
    embedder: Any
    table_name: str

    def _embed(self, text: str) -> list[float]:
        # DSPy Embedder accepts a list[str] and returns a 2D embedding array/list.
        # Convert to native Python floats for LanceDB compatibility.
        return [float(v) for v in self.embedder([text])[0]]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in row] for row in self.embedder(texts)]

    def _infer_embedding_dim(self) -> int:
        if self.table_name in self.db.list_tables():
            table = self.db.open_table(self.table_name)
            dim = get_vector_dim(table)
            if dim is not None:
                return dim
        vec = self._embed("__dimension_probe__")
        return len(vec)
