import logging
import os
import re
from typing import Any

import httpx
import pyarrow as pa
from lancedb.rerankers import Reranker

logger = logging.getLogger("dspy_lancedb_memory")


# Built-in rerank endpoints and the environment variables that carry their keys.
# Providers without an entry require an explicit ``api_base``.
_RERANK_PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
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


class LiteLLMReranker(Reranker):
    """Reranker that calls a Cohere-compatible ``/rerank`` endpoint over HTTP.

    OpenRouter models (``"openrouter/..."`` or names such as
    ``"cohere/rerank-4-fast"``) go to OpenRouter's ``/rerank`` endpoint.
    Other model strings use a built-in provider endpoint (Cohere, Jina,
    Together, Voyage, Fireworks) or ``api_base`` when one is supplied.

    Examples: ``"cohere/rerank-english-v3.0"``,
    ``"jina/jina-reranker-v2-base-multilingual"``, ``"cohere/rerank-4-fast"``.
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

    @property
    def _provider(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else self.model

    @property
    def _model_name(self) -> str:
        """Model id without the routing prefix (``"cohere/model"`` → ``"model"``)."""
        return self.model.split("/", 1)[1] if "/" in self.model else self.model

    @property
    def _uses_openrouter(self) -> bool:
        """OpenRouter is selected by prefix or by Cohere's OpenRouter naming."""
        return self.model.startswith("openrouter/") or bool(
            _OPENROUTER_MODEL.match(self.model)
        )

    def _provider_api_key(self, provider: str) -> str | None:
        if self.api_key:
            return self.api_key
        spec = _RERANK_PROVIDERS.get(provider)
        if spec is None:
            return None
        for env_var in spec[1]:
            value = os.environ.get(env_var)
            if value:
                return value
        return None

    def _rerank(self, result_set: pa.Table, query: str) -> pa.Table:
        result_set = self._handle_empty_results(result_set)
        if len(result_set) == 0:
            return result_set

        docs = result_set[self.column].to_pylist()

        try:
            if self.model.startswith("openrouter/"):
                response = self._rerank_openrouter(query, docs)
            elif self.api_base:
                response = self._rerank_custom_api(query, docs)
            elif self._uses_openrouter:
                response = self._rerank_openrouter(query, docs)
            else:
                response = self._rerank_provider(query, docs)
            indices, scores = self._parse_response(response)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Reranker call failed (%s); returning original results.",
                exc,
            )
            return self._attach_fallback_scores(result_set)

        result_set = result_set.take(indices)
        result_set = result_set.append_column(
            "_relevance_score",
            pa.array(scores, type=pa.float32()),
        )
        return result_set

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> tuple[list[int], list[float]]:
        """Normalize Cohere- and Voyage-shaped responses."""
        rows = response.get("results")
        if rows is None:
            rows = response.get("data") or []
        indices: list[int] = []
        scores: list[float] = []
        for row in rows:
            score = row.get("relevance_score", row.get("score"))
            if score is None:
                continue
            indices.append(row["index"])
            scores.append(score)
        if not indices:
            raise ValueError("rerank response contained no scored results")
        return indices, scores

    def _rerank_provider(self, query: str, docs: list[str]) -> dict[str, Any]:
        """Call the built-in endpoint for the model's provider prefix."""
        provider = self._provider
        spec = _RERANK_PROVIDERS.get(provider)
        if spec is None:
            raise ValueError(
                f"No built-in rerank endpoint for provider {provider!r}. "
                "Pass api_base pointing at a Cohere-compatible /rerank endpoint."
            )
        return self._post_rerank(
            base=spec[0],
            model=self._model_name,
            query=query,
            docs=docs,
            api_key=self._provider_api_key(provider),
        )

    def _rerank_openrouter(self, query: str, docs: list[str]) -> dict[str, Any]:
        """Call OpenRouter's ``/rerank`` endpoint directly."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when using an OpenRouter model"
            )
        # OpenRouter expects ``cohere/rerank-4-fast``, not the routing prefix.
        return self._post_rerank(
            base="https://openrouter.ai/api/v1",
            model=self.model.removeprefix("openrouter/"),
            query=query,
            docs=docs,
            api_key=api_key,
        )

    def _rerank_custom_api(self, query: str, docs: list[str]) -> dict[str, Any]:
        """Call a Cohere-compatible /rerank endpoint on a custom server."""
        assert self.api_base is not None
        return self._post_rerank(
            base=self.api_base,
            model=self._model_name,
            query=query,
            docs=docs,
            api_key=self._provider_api_key(self._provider),
        )

    def _post_rerank(
        self,
        *,
        base: str,
        model: str,
        query: str,
        docs: list[str],
        api_key: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": docs,
        }
        if self.top_n is not None:
            payload["top_n"] = self.top_n

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = httpx.post(
            f"{base.rstrip('/')}/rerank",
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
