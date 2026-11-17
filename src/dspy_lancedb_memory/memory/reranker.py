"""Reranker construction for the ``Store()`` factory."""

from __future__ import annotations

import dspy
from lancedb.rerankers import Reranker

from ..reranking import LiteLLMReranker


def build_reranker(reranker_lm: dspy.LM | str | None) -> Reranker | None:
    """Create a reranker from an LM or model string (``None`` disables)."""
    if reranker_lm is None:
        return None
    model_str = reranker_lm.model if isinstance(reranker_lm, dspy.LM) else reranker_lm
    reranker_kwargs: dict = {}
    if isinstance(reranker_lm, dspy.LM):
        lm_kwargs: dict = getattr(reranker_lm, "kwargs", {})
        if lm_kwargs.get("api_base"):
            reranker_kwargs["api_base"] = lm_kwargs["api_base"]
        if lm_kwargs.get("api_key"):
            reranker_kwargs["api_key"] = lm_kwargs["api_key"]
    return LiteLLMReranker(
        model=model_str,
        column="content",
        **reranker_kwargs,
    )
