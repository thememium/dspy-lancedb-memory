import json
import uuid
from datetime import datetime, timezone
from typing import Any, cast

import dspy
import lancedb
import pyarrow as pa
from lancedb.query import LanceVectorQueryBuilder
from lancedb.rerankers import Reranker

from .extraction import ExtractMemory, MemoryExtractor
from .models import MemoryType, memory_type_from_string


class LanceDSPyMemoryStore:
    def __init__(
        self,
        uri: str = ".lancedb",
        table_name: str = "memories",
        embedding_model: str = "openrouter/openai/text-embedding-3-small",
        embedding_dim: int = 1536,
        signature=None,
        reranker: Reranker | None = None,
        rerank_limit_multiplier: int = 10,
    ):
        self.db = lancedb.connect(uri)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.rerank_limit_multiplier = max(rerank_limit_multiplier, 1)

        self.embedder = dspy.Embedder(
            embedding_model,
            caching=True,
        )
        self.reranker = reranker
        self._extraction_signature = signature or ExtractMemory

        self.table = self._get_or_create_table()

    def _embed(self, text: str) -> list[float]:
        # DSPy Embedder accepts a list[str] and returns a 2D embedding array/list.
        return list(self.embedder([text])[0])

    def _get_or_create_table(self):
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("conversation_id", pa.string()),
                pa.field("memory_type", pa.string()),
                pa.field("content", pa.string()),
                pa.field("metadata", pa.string()),  # JSON-encoded dict
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
            ]
        )

        return self.db.create_table(
            self.table_name,
            data=[
                {
                    "id": "__seed__",
                    "user_id": "__seed__",
                    "conversation_id": "__seed__",
                    "memory_type": "seed",
                    "content": "seed",
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "vector": [0.0] * self.embedding_dim,
                }
            ],
            schema=schema,
        )

    def _build_memory_row(
        self,
        *,
        user_id: str,
        content: str,
        conversation_id: str,
        memory_type: MemoryType | str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        resolved_type = memory_type_from_string(memory_type)
        type_value = (
            resolved_type.value
            if isinstance(resolved_type, MemoryType)
            else resolved_type
        )

        return {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "memory_type": type_value,
            "content": content,
            "metadata": json.dumps(metadata or {}),
            "created_at": now,
            "updated_at": now,
            "vector": self._embed(content),
        }

    @staticmethod
    def _without_vectors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for row in rows:
            clean_row = dict(row)
            clean_row.pop("vector", None)
            if isinstance(clean_row.get("metadata"), str):
                clean_row["metadata"] = json.loads(clean_row["metadata"])
            sanitized.append(clean_row)
        return sanitized

    def create_memories(
        self,
        *,
        user_id: str,
        messages: list[dict[str, str]] | None = None,
        conversation_id: str = "",
        extract: bool = True,
        memory_type: MemoryType | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Store content for a user — either raw or extracted from conversation messages.

        Parameters
        ----------
        user_id : str
            Arbitrary identifier for the user this memory belongs to.
        messages : list[dict[str, str]] | None
            A conversation turn in standard ``{"role": ..., "content": ...}`` format.
            Required when *extract* is ``True`` (default).
        conversation_id : str
            Optional grouping key (e.g. thread / session ID).
        extract : bool
            When ``True``, a DSPy Signature is used to extract **all** salient
            memories from the messages. When ``False``, ``messages`` must contain
            exactly one item which is stored verbatim.
        memory_type : MemoryType | str | None
            Force a specific memory category. If ``None`` while *extract* is ``True``,
            the LLM chooses the categories. If ``None`` while *extract* is ``False``,
            it falls back to ``MemoryType.SEMANTIC``.
        metadata : dict[str, Any] | None
            Bag of structured data attached to the memory row but not embedded.

        Returns
        -------
        list[dict[str, Any]]
            The full rows that were written to LanceDB.
        """
        if extract:
            if not messages:
                raise ValueError("messages are required when extract=True")

            extractor = MemoryExtractor(signature=self._extraction_signature)
            extracted: list[tuple[str, MemoryType | str]] = extractor.forward(
                messages=messages
            )

            stored = [
                self.create_memory(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content=content,
                    memory_type=(
                        memory_type_from_string(memory_type)
                        if memory_type is not None
                        else inferred_type
                    ),
                    metadata=metadata,
                )
                for content, inferred_type in extracted
            ]
            return self._without_vectors(stored)

        if not messages or len(messages) != 1:
            raise ValueError(
                "verbatim (extract=False) requires exactly one message in messages"
            )

        row = self.create_memory(
            user_id=user_id,
            conversation_id=conversation_id,
            content=messages[0]["content"],
            memory_type=(
                memory_type_from_string(memory_type)
                if memory_type is not None
                else MemoryType.SEMANTIC
            ),
            metadata=metadata,
        )
        return self._without_vectors([row])

    def create_memory(
        self,
        *,
        user_id: str,
        content: str,
        conversation_id: str = "",
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._build_memory_row(
            user_id=user_id,
            content=content,
            conversation_id=conversation_id,
            memory_type=memory_type,
            metadata=metadata,
        )
        self.table.add([row])
        return row

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        conversation_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        limit: int = 5,
        use_reranker: bool = False,
    ) -> list[dict[str, Any]]:
        filters = [f"user_id = '{user_id}'"]

        if conversation_id:
            filters.append(f"conversation_id = '{conversation_id}'")

        if memory_type:
            resolved = memory_type_from_string(memory_type)
            type_value = (
                resolved.value if isinstance(resolved, MemoryType) else resolved
            )
            filters.append(f"memory_type = '{type_value}'")

        builder = cast(
            LanceVectorQueryBuilder,
            self.table.search(self._embed(query), vector_column_name="vector"),
        )

        if self.reranker is not None and use_reranker:
            builder = builder.rerank(self.reranker, query_string=query)
            fetch_limit = limit * self.rerank_limit_multiplier
        else:
            fetch_limit = limit

        results = (builder.where(" AND ".join(filters)).limit(fetch_limit).to_list())[
            :limit
        ]
        return self._without_vectors(results)

    def update_memory(self, *, memory_id: str, content: str) -> None:
        self.table.update(
            where=f"id = '{memory_id}'",
            values={
                "content": content,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "vector": self._embed(content),
            },
        )

    def delete_memory(self, *, memory_id: str) -> None:
        self.table.delete(f"id = '{memory_id}'")
