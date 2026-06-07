import json
import logging
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, cast

import dspy
import lancedb
import pyarrow as pa
from lancedb.query import LanceVectorQueryBuilder
from lancedb.rerankers import Reranker

from .extraction import (
    ExtractMemory,
    MemoryExtractor,
    MemoryOperationExtractor,
    MemoryReconciler,
)
from .models import (
    Memories,
    Memory,
    MemoryOperation,
    MemoryType,
    ReconciledMemory,
    memory_type_from_string,
)

logger = logging.getLogger("dspy_lancedb_memory")

_SEMANTIC_DUPLICATE_STOPWORDS = frozenset(
    {"a", "an", "the", "their", "his", "her", "our", "my", "your"}
)


class LanceDSPyMemoryStore:
    _UPSERT_CANDIDATE_LIMIT = 10

    def __init__(
        self,
        uri: str = ".lancedb",
        table_name: str = "memories",
        embedding_lm=None,
        embedding_dim: int | None = None,
        signature=None,
        reranker: Reranker | None = None,
        rerank_limit_multiplier: int = 10,
    ):
        self.db = lancedb.connect(uri)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.rerank_limit_multiplier = max(rerank_limit_multiplier, 1)

        if embedding_lm is None:
            embedding_lm = dspy.LM("openrouter/openai/text-embedding-3-small")
        lm_kwargs: dict = getattr(embedding_lm, "kwargs", {})
        embedder_kwargs: dict = {}
        if lm_kwargs.get("api_base"):
            embedder_kwargs["api_base"] = lm_kwargs["api_base"]
        if lm_kwargs.get("api_key"):
            embedder_kwargs["api_key"] = lm_kwargs["api_key"]
        # When api_base is set the target is an OpenAI-compatible server,
        # so force the ``openai/`` provider prefix regardless of the original
        # model string (e.g. ``huggingface/…``).  LiteLLM's HuggingFace
        # handler builds a provider-specific URL that 404s on custom servers.
        embedder_model = embedding_lm.model
        if "api_base" in embedder_kwargs and "/" in embedder_model:
            embedder_model = "openai/" + embedder_model.split("/", 1)[1]
            embedder_kwargs["encoding_format"] = "float"
        self.embedder = dspy.Embedder(
            embedder_model,
            caching=True,
            **embedder_kwargs,
        )
        self.reranker = reranker
        self._extraction_signature = signature or ExtractMemory

        if self.embedding_dim is None:
            self.embedding_dim = self._infer_embedding_dim()

        self.table = self._get_or_create_table()

    def _embed(self, text: str) -> list[float]:
        # DSPy Embedder accepts a list[str] and returns a 2D embedding array/list.
        # Convert to native Python floats for LanceDB compatibility.
        return [float(v) for v in self.embedder([text])[0]]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in row] for row in self.embedder(texts)]

    def _infer_embedding_dim(self) -> int:
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            dim = self._get_vector_dim(table)
            if dim is not None:
                return dim
        vec = self._embed("__dimension_probe__")
        return len(vec)

    @staticmethod
    def _get_vector_dim(table) -> int | None:
        for field in table.schema:
            if field.name == "vector":
                ft = field.type
                if isinstance(ft, pa.FixedSizeListType):
                    return ft.list_size
                if isinstance(ft, pa.ListType):
                    return None
        return None

    @staticmethod
    def _memory_type_value(memory_type: MemoryType | str | None) -> str | None:
        if memory_type is None:
            return None

        resolved = memory_type_from_string(memory_type)
        return resolved.value if isinstance(resolved, MemoryType) else resolved

    @staticmethod
    def _normalize_semantic_content(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
        tokens = [
            token
            for token in normalized.split()
            if token and token not in _SEMANTIC_DUPLICATE_STOPWORDS
        ]
        return " ".join(tokens)

    @staticmethod
    def _shared_prefix_length(left: list[str], right: list[str]) -> int:
        shared = 0
        for left_token, right_token in zip(left, right, strict=False):
            if left_token != right_token:
                break
            shared += 1
        return shared

    @staticmethod
    def _shared_suffix_length(left: list[str], right: list[str]) -> int:
        shared = 0
        for left_token, right_token in zip(
            reversed(left), reversed(right), strict=False
        ):
            if left_token != right_token:
                break
            shared += 1
        return shared

    @staticmethod
    def _build_filters(
        *,
        user_id: str,
        session_id: str,
        conversation_id: str,
        memory_type: str | None = None,
    ) -> list[str]:
        filters = [f"user_id = '{user_id}'"]

        if session_id:
            filters.append(f"session_id = '{session_id}'")

        if conversation_id:
            filters.append(f"conversation_id = '{conversation_id}'")

        if memory_type:
            filters.append(f"memory_type = '{memory_type}'")

        return filters

    @classmethod
    def _semantic_match_action(
        cls,
        *,
        new_content: str,
        existing_content: str,
        distance: float,
        similarity_threshold: float,
    ) -> str | None:
        new_normalized = cls._normalize_semantic_content(new_content)
        existing_normalized = cls._normalize_semantic_content(existing_content)

        if not new_normalized or not existing_normalized:
            return None

        if new_normalized == existing_normalized:
            return "skip"

        new_tokens = new_normalized.split()
        existing_tokens = existing_normalized.split()
        new_set = set(new_tokens)
        existing_set = set(existing_tokens)
        union = new_set | existing_set
        jaccard = len(new_set & existing_set) / len(union) if union else 0.0
        prefix = cls._shared_prefix_length(new_tokens, existing_tokens)
        suffix = cls._shared_suffix_length(new_tokens, existing_tokens)
        sequence_ratio = SequenceMatcher(
            None, new_normalized, existing_normalized
        ).ratio()
        contains_other = (
            new_normalized in existing_normalized
            or existing_normalized in new_normalized
        )
        cosine_similarity = 1.0 - distance
        same_slot_replacement = prefix >= max(
            2, min(len(new_tokens), len(existing_tokens)) - 1
        )

        same_fact = (
            (prefix >= 2 and jaccard >= 0.5)
            or (suffix >= 2 and jaccard >= 0.5)
            or (contains_other and (prefix >= 1 or suffix >= 1))
            or (same_slot_replacement and sequence_ratio >= 0.88)
        )
        if not same_fact or cosine_similarity < similarity_threshold:
            return None

        new_is_richer = len(new_tokens) > len(existing_tokens) or (
            existing_normalized in new_normalized
            and new_normalized != existing_normalized
        )
        if sequence_ratio >= 0.94 and not new_is_richer and not same_slot_replacement:
            return "skip"

        return "update" if new_is_richer or same_slot_replacement else "skip"

    def _get_or_create_table(self):
        assert self.embedding_dim is not None, (
            "embedding_dim must be set before creating table"
        )
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            existing_dim = self._get_vector_dim(table)
            if existing_dim is not None and existing_dim != self.embedding_dim:
                logger.warning(
                    "Table %r vector dimension (%d) does not match embedding_dim (%d); "
                    "dropping and recreating.",
                    self.table_name,
                    existing_dim,
                    self.embedding_dim,
                )
                self.db.drop_table(self.table_name)
            else:
                return table

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("session_id", pa.string()),
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
                    "session_id": "__seed__",
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
        session_id: str = "",
        conversation_id: str = "",
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
            "session_id": session_id,
            "conversation_id": conversation_id,
            "memory_type": type_value,
            "content": content,
            "metadata": json.dumps(metadata or {}),
            "created_at": now,
            "updated_at": now,
            "vector": self._embed(content),
        }

    @staticmethod
    def _to_memory(row: dict[str, Any]) -> Memory:
        """Convert a raw LanceDB row dict to a ``Memory`` instance."""
        clean = dict(row)
        clean.pop("vector", None)

        # Preserve relevance score: reranker sets _relevance_score directly;
        # otherwise derive cosine similarity from LanceDB's _distance.
        relevance_score = None
        if "_relevance_score" in clean:
            relevance_score = float(clean.pop("_relevance_score"))
        elif "_distance" in clean:
            distance = float(clean.pop("_distance"))
            relevance_score = 1.0 - distance

        if isinstance(clean.get("metadata"), str):
            clean["metadata"] = json.loads(clean["metadata"])

        memory = Memory.model_validate(clean)
        memory.relevance_score = relevance_score
        return memory

    @staticmethod
    def _without_vectors(rows: list[Any]) -> Memories:
        result: list[Memory] = []
        for r in rows:
            if isinstance(r, Memory):
                result.append(r)
            else:
                result.append(LanceDSPyMemoryStore._to_memory(r))
        return result

    def create_memories(
        self,
        *,
        user_id: str,
        contents: list[dict[str, str]] | None = None,
        session_id: str = "",
        conversation_id: str = "",
        extract: bool = True,
        memory_type: MemoryType | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memories:
        """
        Store content for a user — either raw or extracted from conversation.

        Parameters
        ----------
        user_id : str
            Arbitrary identifier for the user this memory belongs to.
        contents : list[dict[str, str]] | None
            A conversation turn in standard ``{"role": ..., "content": ...}`` format.
            Required when *extract* is ``True`` (default).
        session_id : str
            Optional session / thread identifier for grouping.
        conversation_id : str
            Optional grouping key (e.g. conversation / dialogue ID).
        extract : bool
            When ``True``, a DSPy Signature is used to extract **all** salient
            memories from *contents*. When ``False``, *contents* must contain
            exactly one item which is stored verbatim.
        memory_type : MemoryType | str | None
            Force a specific memory category. If ``None`` while *extract* is ``True``,
            the LLM chooses the categories. If ``None`` while *extract* is ``False``,
            it falls back to ``MemoryType.SEMANTIC``.
        metadata : dict[str, Any] | None
            Bag of structured data attached to the memory row but not embedded.

        Returns
        -------
        Memories
            The full rows that were written to LanceDB.
        """
        if extract:
            if not contents:
                raise ValueError("contents is required when extract=True")

            extractor = MemoryExtractor(signature=self._extraction_signature)
            extracted = cast(
                list[tuple[str, MemoryType | str]],
                extractor(messages=contents),
            )

            stored = [
                self.create_memory(
                    user_id=user_id,
                    session_id=session_id,
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

        if not contents or len(contents) != 1:
            raise ValueError(
                "verbatim (extract=False) requires exactly one item in contents"
            )

        row = self.create_memory(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            content=contents[0]["content"],
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
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        row = self._build_memory_row(
            user_id=user_id,
            content=content,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=memory_type,
            metadata=metadata,
        )
        self.table.add([row])
        return self._to_memory(row)

    @staticmethod
    def _filter_by_relevance(
        results: list[dict[str, Any]], min_score: float
    ) -> list[dict[str, Any]]:
        """Drop results whose relevance score falls below *min_score*."""
        filtered: list[dict[str, Any]] = []
        for r in results:
            if "_relevance_score" in r:
                score = float(r["_relevance_score"])
            elif "_distance" in r:
                score = 1.0 - float(r["_distance"])
            else:
                score = 0.0
            if score >= min_score:
                filtered.append(r)
        return filtered

    def search_memories(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str | None = None,
        conversation_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        limit: int = 5,
        use_reranker: bool = False,
        min_relevance_score: float | None = None,
    ) -> Memories:
        if min_relevance_score is None:
            min_relevance_score = (
                0.3 if (self.reranker is not None and use_reranker) else 0.5
            )

        filters = self._build_filters(
            user_id=user_id,
            session_id=session_id or "",
            conversation_id=conversation_id or "",
            memory_type=self._memory_type_value(memory_type),
        )

        builder = cast(
            LanceVectorQueryBuilder,
            self.table.search(self._embed(query), vector_column_name="vector"),
        )

        if self.reranker is not None and use_reranker:
            builder = builder.rerank(self.reranker, query_string=query)
            fetch_limit = limit * self.rerank_limit_multiplier
        else:
            fetch_limit = limit * self.rerank_limit_multiplier

        results = builder.where(" AND ".join(filters)).limit(fetch_limit).to_list()
        results = self._filter_by_relevance(results, min_relevance_score)

        results = results[:limit]
        return self._without_vectors(results)

    def update_memory(self, *, memory_id: str, content: str) -> None:
        old = self.table.search().where(f"id = '{memory_id}'").to_list()
        if not old:
            return
        row = dict(old[0])
        row["content"] = content
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        row["vector"] = self._embed(content)
        self.table.delete(f"id = '{memory_id}'")
        self.table.add([row])

    def upsert_memories(
        self,
        *,
        user_id: str,
        contents: list[dict[str, str]] | None = None,
        session_id: str = "",
        conversation_id: str = "",
        extract: bool = True,
        memory_type: MemoryType | str | None = None,
        metadata: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> Memories:
        """Batch upsert — insert, update, or skip memories based on semantic
        similarity.  Mirrors :meth:`create_memories` but uses
        :meth:`upsert_memory` under the hood.

        When *extract* is ``True`` (default), a DSPy Signature extracts
        salient memories from the conversation turn and each extracted
        memory is independently upserted.

        When *extract* is ``False``, *contents* must contain exactly one
        item which is upserted verbatim.

        Parameters
        ----------
        user_id : str
            User these memories belong to.
        contents : list[dict[str, str]] | None
            A conversation turn in ``{"role": ..., "content": ...}`` format.
            Required when *extract* is ``True``.
        session_id : str
            Optional session scope for matching.
        conversation_id : str
            Optional conversation scope for matching.
        extract : bool
            When ``True``, use DSPy extraction to pull out individual
            memories from the conversation text.
        memory_type : MemoryType | str | None
            Force a specific category.  ``None`` lets the LLM choose when
            extracting; falls back to ``MemoryType.SEMANTIC`` otherwise.
        metadata : dict[str, Any] | None
            Structured data attached to new rows (ignored on update).
        similarity_threshold : float
            Cosine similarity threshold forwarded to :meth:`upsert_memory`.
            Default ``0.85``.
        use_reconciler : bool
            When ``True`` (default), extracted semantic memories are reconciled
            against existing candidates via an LLM-based :class:`MemoryReconciler`
            before any write. Set to ``False`` to skip the LLM and use the fast
            heuristic fallback.

        Returns
        -------
        Memories
            The resulting rows.
        """
        if extract:
            if not contents:
                raise ValueError("contents is required when extract=True")

            extractor = MemoryExtractor(signature=self._extraction_signature)
            extracted = cast(
                list[tuple[str, MemoryType | str]],
                extractor(messages=contents),
            )

            if use_reconciler:
                reconciler = MemoryReconciler()
                results: list[Memory] = []
                for content, inferred_type in extracted:
                    memory_type_str = (
                        memory_type_from_string(memory_type)
                        if memory_type is not None
                        else inferred_type
                    )
                    memory_type_value = self._memory_type_value(memory_type_str)

                    # Retrieve candidates for reconciliation
                    filters = self._build_filters(
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        memory_type=(
                            memory_type_value
                            if memory_type_value == MemoryType.SEMANTIC.value
                            else None
                        ),
                    )
                    candidates = (
                        self.table.search(
                            self._embed(content), vector_column_name="vector"
                        )
                        .where(" AND ".join(filters))
                        .limit(self._UPSERT_CANDIDATE_LIMIT)
                        .to_list()
                    )

                    if candidates:
                        existing_list = [
                            {
                                "id": str(c["id"]),
                                "content": str(c["content"]),
                                "type": str(c["memory_type"]),
                            }
                            for c in candidates
                        ]
                        decision = reconciler(
                            new_memory_content=content,
                            new_memory_type=str(memory_type_str),
                            existing_memories=existing_list,
                        )
                    else:
                        decision = ReconciledMemory(
                            action="create",
                            memory_id="",
                            final_content=content,
                            final_type=str(memory_type_str),
                        )

                    if decision.action == "keep":
                        kept = (
                            self.table.search()
                            .where(f"id = '{decision.memory_id}'")
                            .to_list()
                        )
                        if kept:
                            results.append(self._without_vectors([dict(kept[0])])[0])
                    elif decision.action == "update":
                        self.update_memory(
                            memory_id=decision.memory_id,
                            content=decision.final_content,
                        )
                        updated = (
                            self.table.search()
                            .where(f"id = '{decision.memory_id}'")
                            .to_list()
                        )
                        if updated:
                            results.append(self._without_vectors([dict(updated[0])])[0])
                    else:  # create
                        results.append(
                            self.create_memory(
                                user_id=user_id,
                                content=decision.final_content,
                                session_id=session_id,
                                conversation_id=conversation_id,
                                memory_type=memory_type_str,
                                metadata=metadata,
                            )
                        )

                return results

            # Fast fallback: delegate each extracted memory to upsert_memory
            stored = [
                self.upsert_memory(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    content=content,
                    memory_type=(
                        memory_type_from_string(memory_type)
                        if memory_type is not None
                        else inferred_type
                    ),
                    metadata=metadata,
                    similarity_threshold=similarity_threshold,
                    use_reconciler=False,
                )
                for content, inferred_type in extracted
            ]
            return stored  # upsert_memory already strips vectors

        if not contents or len(contents) != 1:
            raise ValueError(
                "verbatim (extract=False) requires exactly one item in contents"
            )

        row = self.upsert_memory(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            content=contents[0]["content"],
            memory_type=(
                memory_type_from_string(memory_type)
                if memory_type is not None
                else MemoryType.SEMANTIC
            ),
            metadata=metadata,
            similarity_threshold=similarity_threshold,
            use_reconciler=use_reconciler,
        )
        return [row]

    def upsert_memory(
        self,
        *,
        user_id: str,
        content: str,
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> Memory:
        """Insert or update a memory based on semantic similarity.

        Three-way decision:

        1. **Exact match** — If a memory with the same *content* string
           already exists for this user, skip the write and return the
           existing row unchanged (no-op).
        2. **Semantic match** — If no exact match but one of the nearest
           scoped candidates is judged to represent the same semantic fact,
           update that memory in place with the refined or corrected content.
        3. **No match** — No existing memory is close enough; insert a new
           row.

        Parameters
        ----------
        user_id : str
            User the memory belongs to.
        content : str
            New (or updated) memory text.
        session_id : str
            Optional session scope for the match.
        conversation_id : str
            Optional conversation scope for the match.
        memory_type : MemoryType | str
            Category for the memory.  Only used when creating new rows.
        metadata : dict[str, Any] | None
            Bag of structured data.  Only used when creating new rows.
        similarity_threshold : float
            Minimum cosine similarity (0‑1) to consider two memories
            semantically equivalent.  Default ``0.85``.
        use_reconciler : bool
            When ``True`` (default) and the memory type is ``"semantic"``,
            candidates are fed to an LLM-based :class:`MemoryReconciler` to
            decide whether to keep, update, or create. Set to ``False`` to
            use the fast heuristic fallback instead.

        Returns
        -------
        Memory
            The resulting memory row.
        """
        resolved_type = self._memory_type_value(memory_type)
        candidate_limit = self._UPSERT_CANDIDATE_LIMIT

        filters = self._build_filters(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=(
                resolved_type if resolved_type == MemoryType.SEMANTIC.value else None
            ),
        )
        results = (
            self.table.search(self._embed(content), vector_column_name="vector")
            .where(" AND ".join(filters))
            .limit(candidate_limit)
            .to_list()
        )

        if results:
            if resolved_type == MemoryType.SEMANTIC.value:
                if use_reconciler:
                    # Use the LLM reconciler for nuanced keep/update/create
                    # decisions among the semantic candidate pool.
                    reconciler = MemoryReconciler()
                    existing_list = [
                        {
                            "id": str(c["id"]),
                            "content": str(c["content"]),
                            "type": str(c["memory_type"]),
                        }
                        for c in results
                    ]
                    decision = reconciler(
                        new_memory_content=content,
                        new_memory_type=str(resolved_type),
                        existing_memories=existing_list,
                    )

                    if decision.action == "keep":
                        kept = (
                            self.table.search()
                            .where(f"id = '{decision.memory_id}'")
                            .to_list()
                        )
                        if kept:
                            return self._without_vectors([dict(kept[0])])[0]
                    elif decision.action == "update":
                        self.update_memory(
                            memory_id=decision.memory_id,
                            content=decision.final_content,
                        )
                        updated = (
                            self.table.search()
                            .where(f"id = '{decision.memory_id}'")
                            .to_list()
                        )
                        if updated:
                            return self._without_vectors([dict(updated[0])])[0]
                else:
                    for candidate in results:
                        existing = dict(candidate)
                        action = self._semantic_match_action(
                            new_content=content,
                            existing_content=str(existing["content"]),
                            distance=float(existing.get("_distance", 1.0)),
                            similarity_threshold=similarity_threshold,
                        )
                        if action == "skip":
                            return self._without_vectors([existing])[0]
                        if action == "update":
                            self.update_memory(
                                memory_id=str(existing["id"]), content=content
                            )
                            updated = (
                                self.table.search()
                                .where(f"id = '{existing['id']}'")
                                .to_list()
                            )
                            return self._without_vectors([dict(updated[0])])[0]
            else:
                for candidate in results:
                    existing = dict(candidate)
                    if existing["content"] == content:
                        return self._without_vectors([existing])[0]

                existing = dict(results[0])
                distance = float(existing.get("_distance", 1.0))
                cosine_sim = 1.0 - distance

                if cosine_sim >= similarity_threshold:
                    self.update_memory(memory_id=str(existing["id"]), content=content)
                    updated = (
                        self.table.search().where(f"id = '{existing['id']}'").to_list()
                    )
                    return self._without_vectors([dict(updated[0])])[0]

        # --- No match → create ---
        return self._without_vectors(
            [
                self.create_memory(
                    user_id=user_id,
                    content=content,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=memory_type,
                    metadata=metadata,
                )
            ]
        )[0]

    def delete_memory(self, *, memory_id: str) -> None:
        self.table.delete(f"id = '{memory_id}'")

    def delete_memories_by_search(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str | None = None,
        similarity_threshold: float = 0.85,
        limit: int = 5,
    ) -> Memories:
        filters = self._build_filters(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=self._memory_type_value(memory_type),
        )

        candidates = (
            self.table.search(self._embed(query), vector_column_name="vector")
            .where(" AND ".join(filters))
            .limit(limit)
            .to_list()
        )

        if not candidates:
            return []

        results = self._filter_by_relevance(candidates, min_score=similarity_threshold)

        if not results:
            return []

        deleted = []
        for row in results:
            memory_id = str(row["id"])
            self.table.delete(f"id = '{memory_id}'")
            deleted.append(self._to_memory(dict(row)))

        return deleted

    def process_memories(
        self,
        *,
        user_id: str,
        contents: list[dict[str, str]] | None = None,
        session_id: str = "",
        conversation_id: str = "",
        extract: bool = True,
        metadata: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> tuple[Memories, Memories]:
        """Extract and process memory operations (create/update/delete) from a conversation.

        Returns a tuple of (created_or_updated_memories, deleted_memories).
        """
        if not contents:
            raise ValueError("contents is required")

        if extract:
            extractor = MemoryOperationExtractor()
            operations = extractor(messages=contents)
        else:
            operations = [
                MemoryOperation(
                    action="create",
                    content=c["content"],
                )
                for c in contents
            ]

        created_or_updated: list[Memory] = []
        deleted: list[Memory] = []

        for op in operations:
            action = op.action.strip().lower()

            if action == "delete":
                search_query = op.search_query or op.content
                if not search_query:
                    continue

                if op.content:
                    exact_filters = self._build_filters(
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        memory_type=self._memory_type_value(op.memory_type),
                    )
                    exact_results = (
                        self.table.search()
                        .where(
                            " AND ".join(exact_filters)
                            + f" AND content = '{op.content}'"
                        )
                        .limit(1)
                        .to_list()
                    )
                    if exact_results:
                        for row in exact_results:
                            memory_id = str(row["id"])
                            self.table.delete(f"id = '{memory_id}'")
                            deleted.append(self._to_memory(dict(row)))
                        continue

                removed = self.delete_memories_by_search(
                    user_id=user_id,
                    query=search_query,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=op.memory_type or None,
                    similarity_threshold=similarity_threshold,
                    limit=3,
                )
                deleted.extend(removed)

            elif action == "update":
                result = self.upsert_memory(
                    user_id=user_id,
                    content=op.content,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=op.memory_type or MemoryType.SEMANTIC,
                    metadata=metadata,
                    similarity_threshold=similarity_threshold,
                    use_reconciler=use_reconciler,
                )
                created_or_updated.append(result)

            elif action == "create":
                result = self.create_memory(
                    user_id=user_id,
                    content=op.content,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=op.memory_type or MemoryType.SEMANTIC,
                    metadata=metadata,
                )
                created_or_updated.append(result)

        return (created_or_updated, deleted)
