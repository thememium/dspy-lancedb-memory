import logging
import os
from typing import Any

import dspy
import httpx
import pyarrow as pa
from lancedb.rerankers import Reranker

logger = logging.getLogger("dspy_memory")


class OpenRouterReranker(Reranker):
    """Reranker that calls the OpenRouter /rerank endpoint (Cohere-compatible).

    Accepts a ``dspy.LM`` directly so callers don't need to handle model
    strings or API keys separately.
    """

    def __init__(
        self,
        model: str = "cohere/rerank-4-fast",
        column: str = "text",
        top_n: int | None = None,
        api_key: str | None = None,
        lm: dspy.LM | None = None,
        return_score: str = "relevance",
    ):
        super().__init__(return_score)
        self.column = column
        self.top_n = top_n
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if lm is not None:
            self.model = lm.model
            # The ``dspy.LM`` stores the provider prefix (e.g. ``openrouter/cohere/...``)
            # but the OpenRouter /rerank endpoint expects the bare model name
            # (e.g. ``cohere/rerank-4-fast``).  Strip the known prefix.
            if "/" in (lm._provider_name or ""):
                # multi-part provider → assume the model string is already bare
                pass
            elif self.model.startswith(f"{lm._provider_name}/"):
                self.model = self.model[len(lm._provider_name) + 1 :]
            # Fall back to api_key from the LM's kwargs if not set otherwise
            if self.api_key is None and hasattr(lm, "kwargs"):
                self.api_key = lm.kwargs.get("api_key") or lm.kwargs.get("api_key")
        else:
            self.model = model

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Pass api_key=... or set the env var."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rerank(self, result_set: pa.Table, query: str) -> pa.Table:
        result_set = self._handle_empty_results(result_set)
        if len(result_set) == 0:
            return result_set

        docs = result_set[self.column].to_pylist()
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": docs,
        }
        if self.top_n is not None:
            payload["top_n"] = self.top_n

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "Reranker call failed (%s); returning original results.", exc
            )
            return result_set

        data = response.json()
        results = data["results"]
        indices, scores = zip(*[(r["index"], r["relevance_score"]) for r in results])
        result_set = result_set.take(list(indices))
        result_set = result_set.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )
        return result_set

    # ------------------------------------------------------------------
    # LanceDB reranker interface
    # ------------------------------------------------------------------

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        vector_results = self._rerank(vector_results, query)
        if self.score == "relevance":
            vector_results = vector_results.drop_columns(["_distance"])
        return vector_results

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        fts_results = self._rerank(fts_results, query)
        if self.score == "relevance":
            fts_results = fts_results.drop_columns(["_score"])
        return fts_results

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        if self.score == "all":
            combined = self._merge_and_keep_scores(vector_results, fts_results)
        else:
            combined = self.merge_results(vector_results, fts_results)
        combined = self._rerank(combined, query)
        if self.score == "relevance":
            combined = self._keep_relevance_score(combined)
        return combined
