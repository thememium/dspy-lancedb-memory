"""Memory retrieval: vector search, hybrid fusion, and reranking."""

from __future__ import annotations

from typing import Any, cast

import pyarrow as pa
from lancedb.query import LanceVectorQueryBuilder

from ..models import Memories, MemoryType, ScopeLike, memory_type_value
from .filters import build_filters, filter_rows_by_json_fields
from .rows import without_vectors
from .similarity import distance_to_similarity


def rrf_rows(
    vector_rows: list[dict[str, Any]],
    fts_rows: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion over rows keyed by ``id``."""
    scores: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    for ranked_rows in (vector_rows, fts_rows):
        for rank, row in enumerate(ranked_rows):
            key = str(row["id"])
            rows.setdefault(key, row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    merged: list[dict[str, Any]] = []
    for key in sorted(scores, key=scores.__getitem__, reverse=True):
        row = dict(rows[key])
        row.pop("_distance", None)
        row.pop("_score", None)
        row["_relevance_score"] = scores[key]
        merged.append(row)
    return merged


def filter_by_relevance(
    results: list[dict[str, Any]], min_score: float
) -> list[dict[str, Any]]:
    """Drop results whose relevance score falls below *min_score*."""
    filtered: list[dict[str, Any]] = []
    for result in results:
        if "_relevance_score" in result:
            score = float(result["_relevance_score"])
        elif "_distance" in result:
            score = distance_to_similarity(float(result["_distance"]))
        else:
            score = 0.0
        if score >= min_score:
            filtered.append(result)
    return filtered


class SearchMixin:
    """Search methods for the memory store."""

    table: Any
    reranker: Any
    rerank_limit_multiplier: int

    _embed: Any

    def _select_all(self, builder, *extra_columns: str):
        """Project every table column plus the requested score column."""
        return builder.select([*self.table.schema.names, *extra_columns])

    def _hybrid_results(
        self,
        *,
        query: str,
        filters: list[str],
        fetch_limit: int,
        refine_factor: int | None,
        use_reranker: bool,
    ) -> list[dict[str, Any]]:
        """Vector + FTS search merged locally (RTE rejects server-side hybrid)."""
        where = " AND ".join(filters)
        vector_builder = cast(
            LanceVectorQueryBuilder,
            self._select_all(
                self.table.search(self._embed(query), vector_column_name="vector"),
                "_distance",
            ),
        )
        if refine_factor is not None:
            vector_builder = vector_builder.refine_factor(refine_factor)
        vector_rows = [
            dict(row)
            for row in vector_builder.where(where).limit(fetch_limit).to_list()
        ]
        fts_rows = [
            dict(row)
            for row in self._select_all(
                self.table.search(query, query_type="fts", fts_columns="content"),
                "_score",
            )
            .where(where)
            .limit(fetch_limit)
            .to_list()
        ]

        if use_reranker and self.reranker is not None and vector_rows and fts_rows:
            vector_table = pa.Table.from_pylist(
                [{**row, "_rowid": row["id"]} for row in vector_rows]
            )
            fts_table = pa.Table.from_pylist(
                [{**row, "_rowid": row["id"]} for row in fts_rows]
            )
            ranked = self.reranker.rerank_hybrid(query, vector_table, fts_table)
            rows = [dict(row) for row in ranked.to_pylist()]
            for row in rows:
                row.pop("_rowid", None)
            return rows

        return rrf_rows(vector_rows, fts_rows)

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None = None,
        conversation_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        scope: ScopeLike | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 5,
        use_reranker: bool = False,
        min_relevance_score: float | None = None,
        query_type: str = "vector",
        refine_factor: int | None = None,
    ) -> Memories:
        if min_relevance_score is None:
            if self.reranker is not None and use_reranker:
                min_relevance_score = 0.3
            elif query_type == "hybrid":
                min_relevance_score = 0.0
            else:
                min_relevance_score = 0.5

        filters = build_filters(
            user_id=user_id,
            session_id=session_id or "",
            conversation_id=conversation_id or "",
            memory_type=memory_type_value(memory_type),
        )

        fetch_limit = limit * self.rerank_limit_multiplier
        if scope or metadata_filter:
            fetch_limit *= 10

        if query_type == "hybrid":
            # LanceDB's hybrid builder requests row ids internally, which the
            # RTE scanner rejects. Run both searches and merge locally instead.
            results = self._hybrid_results(
                query=query,
                filters=filters,
                fetch_limit=fetch_limit,
                refine_factor=refine_factor,
                use_reranker=use_reranker,
            )
        else:
            builder = cast(
                LanceVectorQueryBuilder,
                self._select_all(
                    self.table.search(self._embed(query), vector_column_name="vector"),
                    "_distance",
                ),
            )
            if refine_factor is not None:
                builder = builder.refine_factor(refine_factor)
            if self.reranker is not None and use_reranker:
                builder = builder.rerank(self.reranker, query_string=query)
            results = builder.where(" AND ".join(filters)).limit(fetch_limit).to_list()
        results = filter_rows_by_json_fields(
            [dict(row) for row in results],
            scope=scope,
            metadata_filter=metadata_filter,
        )
        results = filter_by_relevance(results, min_relevance_score)

        results = results[:limit]
        return without_vectors(results)
