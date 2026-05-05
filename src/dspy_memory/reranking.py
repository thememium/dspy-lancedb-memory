import logging
from typing import Any

import pyarrow as pa
from lancedb.rerankers import Reranker
from litellm import rerank

logger = logging.getLogger("dspy_memory")


class LiteLLMReranker(Reranker):
    """Reranker that uses ``litellm.rerank()`` for cross-encoder reranking.

    Supports any provider with a rerank endpoint (Cohere, Jina, OpenRouter,
    etc.) through LiteLLM's model routing — just use the standard
    ``"provider/model"`` format (e.g. ``"cohere/rerank-english-v3.0"``,
    ``"openrouter/cohere/rerank-4-fast"``).
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
