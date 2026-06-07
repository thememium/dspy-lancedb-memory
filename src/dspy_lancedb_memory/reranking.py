import logging
import os
from typing import Any

import httpx
import pyarrow as pa
from lancedb.rerankers import Reranker
from litellm import rerank

logger = logging.getLogger("dspy_lancedb_memory")


_LITELLM_RERANK_PROVIDERS = frozenset(
    {
        "cohere",
        "together_ai",
        "azure_ai",
        "infinity",
        "litellm_proxy",
        "hosted_vllm",
        "deepinfra",
        "fireworks_ai",
        "voyage",
        "watsonx",
    }
)


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
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(return_score)
        self.model = model
        self.column = column
        self.top_n = top_n
        self.api_base = api_base
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rerank(self, result_set: pa.Table, query: str) -> pa.Table:
        result_set = self._handle_empty_results(result_set)
        if len(result_set) == 0:
            return result_set

        docs = result_set[self.column].to_pylist()

        try:
            provider = self.model.split("/", 1)[0] if "/" in self.model else "openai"
            use_litellm = self.model.startswith("openrouter/") is False and (
                not self.api_base or provider in _LITELLM_RERANK_PROVIDERS
            )
            if self.model.startswith("openrouter/"):
                response = self._rerank_openrouter(query, docs)
            elif use_litellm:
                rerank_kwargs: dict = {}
                if self.api_base:
                    rerank_kwargs["api_base"] = self.api_base
                if self.api_key:
                    rerank_kwargs["api_key"] = self.api_key
                response: Any = rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                    top_n=self.top_n,
                    **rerank_kwargs,
                )
            else:
                response = self._rerank_custom_api(query, docs)
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

    def _rerank_custom_api(self, query: str, docs: list[str]) -> dict[str, Any]:
        """Call a Cohere-compatible /rerank endpoint on a custom server."""
        # Strip provider prefix (e.g. "huggingface/model" → "model").
        model_name = self.model.split("/", 1)[1] if "/" in self.model else self.model
        payload: dict[str, Any] = {
            "model": model_name,
            "query": query,
            "documents": docs,
        }
        if self.top_n is not None:
            payload["top_n"] = self.top_n

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        base = (self.api_base or "").rstrip("/")
        response = httpx.post(
            f"{base}/rerank",
            headers=headers,
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
