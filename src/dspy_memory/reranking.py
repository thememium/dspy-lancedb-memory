import logging
import os
from typing import Any

import httpx
import pyarrow as pa
from lancedb.rerankers import Reranker
from litellm import rerank

logger = logging.getLogger("dspy_memory")


class LiteLLMReranker(Reranker):
    """Reranker that uses ``litellm.rerank()`` for cross-encoder reranking.

    When the model string starts with ``openrouter/`` the class makes a
    direct HTTP call to OpenRouter's ``/rerank`` endpoint instead (since
    LiteLLM does not support OpenRouter as a rerank provider).  All other
    model strings are passed through to ``litellm.rerank()``.

    Examples: ``"cohere/rerank-english-v3.0"``, ``"openrouter/cohere/rerank-4-fast"``.
    """

    def __init__(
        self,
        model: str = "cohere/rerank-english-v3.0",
        column: str = "text",
        top_n: int | None = None,
        return_score: str = "relevance",
    ):
        super().__init__(return_score)
        self.model = model
        self.column = column
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rerank(self, result_set: pa.Table, query: str) -> pa.Table:
        result_set = self._handle_empty_results(result_set)
        if len(result_set) == 0:
            return result_set

        docs = result_set[self.column].to_pylist()

        try:
            if self.model.startswith("openrouter/"):
                response = self._rerank_openrouter(query, docs)
            else:
                response: Any = rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                    top_n=self.top_n,
                )
        except Exception as exc:
            logger.warning(
                "Reranker call failed (%s); returning original results.",
                exc,
            )
            return self._attach_fallback_scores(result_set)

        results = response["results"]
        indices, scores = zip(*[(r["index"], r["relevance_score"]) for r in results])
        result_set = result_set.take(list(indices))
        result_set = result_set.append_column(
            "_relevance_score",
            pa.array(scores, type=pa.float32()),
        )
        return result_set

    def _rerank_openrouter(self, query: str, docs: list[str]) -> dict[str, Any]:
        """Call OpenRouter's ``/rerank`` endpoint directly."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when using an openrouter/ model"
            )
        # OpenRouter expects the bare model name without the provider prefix.
        payload: dict[str, Any] = {
            "model": self.model[len("openrouter/") :],
            "query": query,
            "documents": docs,
        }
        if self.top_n is not None:
            payload["top_n"] = self.top_n

        response = httpx.post(
            "https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    def _attach_fallback_scores(self, result_set: pa.Table) -> pa.Table:
        """Add ``_relevance_score`` so LanceDB's post-rerank validation passes."""
        if "_distance" in result_set.column_names:
            dist = result_set["_distance"].to_pylist()
            scores = [1.0 / (1.0 + d) for d in dist]
        else:
            scores = [0.0] * len(result_set)
        return result_set.append_column(
            "_relevance_score",
            pa.array(scores, type=pa.float32()),
        )

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
