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
    PendingReconciliation,
    ReconciledMemory,
    memory_type_from_string,
)

logger = logging.getLogger("dspy_lancedb_memory")

_SEMANTIC_DUPLICATE_STOPWORDS = frozenset(
    {"a", "an", "the", "their", "his", "her", "our", "my", "your"}
)


class _SingleMemoryReconciler:
    """Callable wrapper for dspy.Parallel — reconciles one memory against the store."""

    def __init__(
        self,
        store: "LanceDSPyMemoryStore",
        reconciler: MemoryReconciler,
        memory_type: MemoryType | str | None,
        skip_threshold: float,
    ):
        self._store = store
        self._reconciler = reconciler
        self._memory_type = memory_type
        self._skip_threshold = skip_threshold

    def __call__(self, **kwargs: Any) -> PendingReconciliation:
        return self._store._reconcile_one(
            content=kwargs["content"],
            inferred_type=kwargs["inferred_type"],
            memory_type=self._memory_type,
            user_id=kwargs["user_id"],
            session_id=kwargs["session_id"],
            conversation_id=kwargs["conversation_id"],
            scope=kwargs["scope"],
            metadata=kwargs["metadata"],
            skip_threshold=self._skip_threshold,
            reconciler=self._reconciler,
        )


class LanceDSPyMemoryStore:
    _UPSERT_CANDIDATE_LIMIT = 10

    def __init__(
        self,
        uri: str = ".lancedb",
        table_name: str = "memories",
        extraction_lm: dspy.LM | None = None,
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
        self._extraction_lm = extraction_lm

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

    _SCOPE_METADATA_KEY = "_scope"

    @staticmethod
    def _json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if not value:
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @classmethod
    def _split_metadata_scope(
        cls, metadata: dict[str, Any] | str | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        clean_metadata = cls._json_dict(metadata)
        scope = cls._json_dict(clean_metadata.pop(cls._SCOPE_METADATA_KEY, {}))
        return clean_metadata, scope

    @classmethod
    def _pack_metadata(
        cls,
        metadata: dict[str, Any] | None,
        scope: dict[str, Any] | None,
    ) -> str:
        packed = dict(metadata or {})
        if scope:
            packed[cls._SCOPE_METADATA_KEY] = dict(scope)
        return json.dumps(packed)

    @staticmethod
    def _matches_filter(value: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                return False
            return all(
                key in value
                and LanceDSPyMemoryStore._matches_filter(value[key], nested)
                for key, nested in expected.items()
            )
        if isinstance(expected, (list, tuple, set)):
            return value in expected
        return value == expected

    @classmethod
    def _row_matches_json_filters(
        cls,
        row: dict[str, Any],
        *,
        scope: dict[str, Any] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> bool:
        metadata, row_scope = cls._split_metadata_scope(row.get("metadata"))
        if scope and not cls._matches_filter(row_scope, scope):
            return False
        if metadata_filter and not cls._matches_filter(metadata, metadata_filter):
            return False
        return True

    @classmethod
    def _filter_rows_by_json_fields(
        cls,
        rows: list[dict[str, Any]],
        *,
        scope: dict[str, Any] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not scope and not metadata_filter:
            return rows
        return [
            row
            for row in rows
            if cls._row_matches_json_filters(
                row, scope=scope, metadata_filter=metadata_filter
            )
        ]

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

        filters.append("is_active = true")

        return filters

    @classmethod
    def _semantic_match_action(
        cls,
        *,
        new_content: str,
        existing_content: str,
        distance: float,
        similarity_threshold: float,
        skip_threshold: float = 0.85,
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
        new_is_richer = len(new_tokens) > len(existing_tokens) or (
            existing_normalized in new_normalized
            and new_normalized != existing_normalized
        )

        # --- Skip-threshold gate ------------------------------------------------
        # When cosine similarity is very high, the vectors are near-identical.
        # Skip the write unless the new content is genuinely richer or replaces
        # a specific slot value (e.g. "color is blue" → "color is red").
        # This prevents runaway duplicate memories across repeated upsert runs.
        if cosine_similarity >= skip_threshold:
            if not new_is_richer and not same_slot_replacement:
                return "skip"

        same_fact = (
            (prefix >= 2 and jaccard >= 0.5)
            or (suffix >= 2 and jaccard >= 0.5)
            or (contains_other and (prefix >= 1 or suffix >= 1))
            or (same_slot_replacement and sequence_ratio >= 0.88)
        )
        if not same_fact or cosine_similarity < similarity_threshold:
            return None

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
                pa.field("replaces_id", pa.utf8(), nullable=True),
                pa.field("is_active", pa.bool_()),
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
                    "replaces_id": None,
                    "is_active": True,
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
        scope: dict[str, Any] | None = None,
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
            "metadata": self._pack_metadata(metadata, scope),
            "created_at": now,
            "updated_at": now,
            "replaces_id": None,
            "is_active": True,
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

        metadata, scope = LanceDSPyMemoryStore._split_metadata_scope(
            clean.get("metadata")
        )
        clean["metadata"] = metadata
        clean["scope"] = scope

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

    @staticmethod
    def _merge_metadata(
        base: dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(base or {})
        merged.update(extra or {})
        return merged

    @staticmethod
    def _normalize_extracted_memories(
        extracted: list[Any],
    ) -> list[tuple[str, MemoryType | str, dict[str, Any]]]:
        normalized: list[tuple[str, MemoryType | str, dict[str, Any]]] = []
        seen_contents: set[str] = set()
        for item in extracted:
            item_metadata: dict[str, Any] = {}
            if isinstance(item, tuple):
                if len(item) == 2:
                    content, inferred_type = item
                elif len(item) >= 3:
                    content, inferred_type, item_metadata = item[:3]
                else:
                    continue
            else:
                content = getattr(item, "content", "")
                inferred_type = getattr(item, "type", MemoryType.SEMANTIC)
                item_metadata = dict(getattr(item, "metadata", {}) or {})
            content_value = str(content).strip()
            if not content_value or content_value in seen_contents:
                continue
            seen_contents.add(content_value)
            normalized.append(
                (
                    content_value,
                    memory_type_from_string(inferred_type),
                    dict(item_metadata or {}),
                )
            )
        return normalized

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
        scope: dict[str, Any] | None = None,
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
            exactly one item and is stored verbatim.
        memory_type : MemoryType | str | None
            Force a specific memory category. If ``None`` while *extract* is ``True``,
            the LLM chooses categories. If ``None`` while *extract* is ``False``,
            falls back to ``MemoryType.SEMANTIC``.
        metadata : dict[str, Any] | None
            Structured data attached to every new memory row but not embedded.
        scope : dict[str, Any] | None
            Custom ownership/query dimensions attached to every new memory row.

        Returns
        -------
        Memories
            The full rows written to LanceDB.
        """
        if extract:
            if not contents:
                raise ValueError("contents is required when extract=True")

            extractor = MemoryExtractor(signature=self._extraction_signature)
            with dspy.context(lm=self._extraction_lm):
                extracted = self._normalize_extracted_memories(
                    cast(list[Any], extractor(messages=contents).memories)
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
                    metadata=self._merge_metadata(metadata, item_metadata),
                    scope=scope,
                )
                for content, inferred_type, item_metadata in extracted
            ]
            return self._without_vectors(stored)

        if not contents or len(contents) != 1:
            raise ValueError(
                "verbatim storage (extract=False) requires exactly one item in contents"
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
            scope=scope,
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
        scope: dict[str, Any] | None = None,
    ) -> Memory:
        row = self._build_memory_row(
            user_id=user_id,
            content=content,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=memory_type,
            metadata=metadata,
            scope=scope,
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
        scope: dict[str, Any] | None = None,
        metadata_filter: dict[str, Any] | None = None,
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

        fetch_limit = limit * self.rerank_limit_multiplier
        if scope or metadata_filter:
            fetch_limit *= 10

        if self.reranker is not None and use_reranker:
            builder = builder.rerank(self.reranker, query_string=query)

        results = builder.where(" AND ".join(filters)).limit(fetch_limit).to_list()
        results = self._filter_rows_by_json_fields(
            [dict(row) for row in results],
            scope=scope,
            metadata_filter=metadata_filter,
        )
        results = self._filter_by_relevance(results, min_relevance_score)

        results = results[:limit]
        return self._without_vectors(results)

    def update_memory(self, *, memory_id: str, content: str) -> None:
        old = self.table.search().where(f"id = '{memory_id}'").to_list()
        if not old:
            return
        old_record = old[0]
        if not old_record.get("is_active", True):
            logger.warning(
                "update_memory called on inactive record %s — skipping", memory_id
            )
            return

        metadata, scope = self._split_metadata_scope(old_record.get("metadata"))
        new_row = self._build_memory_row(
            user_id=old_record["user_id"],
            content=content,
            session_id=old_record.get("session_id", ""),
            conversation_id=old_record.get("conversation_id", ""),
            memory_type=old_record.get("memory_type", "semantic"),
            metadata=metadata,
            scope=scope,
        )
        new_row["replaces_id"] = memory_id

        self.table.add([new_row])
        self.table.update(
            where=f"id = '{memory_id}'",
            values={"is_active": False},
        )

    def _reconcile_one(
        self,
        content: str,
        inferred_type: MemoryType | str,
        memory_type: MemoryType | str | None,
        user_id: str,
        session_id: str,
        conversation_id: str,
        scope: dict[str, Any] | None,
        metadata: dict[str, Any],
        skip_threshold: float,
        reconciler: MemoryReconciler,
    ) -> PendingReconciliation:
        memory_type_str = (
            memory_type_from_string(memory_type)
            if memory_type is not None
            else inferred_type
        )
        memory_type_value = self._memory_type_value(memory_type_str)

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
        candidate_limit = self._UPSERT_CANDIDATE_LIMIT
        fetch_limit = candidate_limit * (10 if scope else 1)
        candidates = (
            self.table.search(self._embed(content), vector_column_name="vector")
            .where(" AND ".join(filters))
            .limit(fetch_limit)
            .to_list()
        )
        candidates = self._filter_rows_by_json_fields(
            [dict(row) for row in candidates], scope=scope
        )[:candidate_limit]

        if candidates:
            nearest_distance = float(candidates[0].get("_distance", 1.0))
            nearest_similarity = 1.0 - nearest_distance
            if nearest_similarity >= skip_threshold:
                decision = ReconciledMemory(
                    action="keep",
                    memory_id=str(candidates[0]["id"]),
                    final_content=str(candidates[0]["content"]),
                    final_type=str(candidates[0]["memory_type"]),
                )
                return PendingReconciliation(
                    content=content,
                    inferred_type=str(memory_type_str),
                    decision=decision,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    scope=dict(scope or {}),
                    metadata=metadata,
                    existing_row=self._without_vectors([dict(candidates[0])])[0],
                )

            existing_list = [
                {
                    "id": str(c["id"]),
                    "content": str(c["content"]),
                    "type": str(c["memory_type"]),
                }
                for c in candidates
            ]
            with dspy.context(lm=self._extraction_lm):
                decision = reconciler(
                    new_memory_content=content,
                    new_memory_type=str(memory_type_str),
                    existing_memories=existing_list,
                ).reconciled
        else:
            decision = ReconciledMemory(
                action="create",
                memory_id="",
                final_content=content,
                final_type=str(memory_type_str),
            )

        return PendingReconciliation(
            content=content,
            inferred_type=str(memory_type_str),
            decision=decision,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            scope=dict(scope or {}),
            metadata=metadata,
        )

    def _reconcile_batch_parallel(
        self,
        extracted: list[tuple[str, MemoryType | str, dict[str, Any]]],
        reconciler: MemoryReconciler,
        user_id: str,
        session_id: str,
        conversation_id: str,
        memory_type: MemoryType | str | None,
        scope: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        skip_threshold: float,
        num_threads: int,
    ) -> list[PendingReconciliation]:
        exec_pairs: list[tuple[Any, dspy.Example]] = []
        for content, inferred_type, item_metadata in extracted:
            module = _SingleMemoryReconciler(
                store=self,
                reconciler=reconciler,
                memory_type=memory_type,
                skip_threshold=skip_threshold,
            )
            example = dspy.Example(
                content=content,
                inferred_type=inferred_type,
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                scope=dict(scope or {}),
                metadata=self._merge_metadata(metadata, item_metadata),
            ).with_inputs(
                "content",
                "inferred_type",
                "user_id",
                "session_id",
                "conversation_id",
                "scope",
                "metadata",
            )
            exec_pairs.append((module, example))

        parallel = dspy.Parallel(
            num_threads=num_threads,
            return_failed_examples=False,
            disable_progress_bar=True,
        )
        return list(parallel(exec_pairs))

    def _apply_reconciliations(
        self, pending: list[PendingReconciliation]
    ) -> list[Memory]:
        results: list[Memory] = []
        for p in pending:
            if p.decision.action == "keep":
                if hasattr(p, "existing_row") and p.existing_row:
                    results.append(p.existing_row)
                else:
                    kept = (
                        self.table.search()
                        .where(f"id = '{p.decision.memory_id}'")
                        .to_list()
                    )
                    if kept:
                        results.append(self._without_vectors([dict(kept[0])])[0])
            elif p.decision.action == "update":
                self.update_memory(
                    memory_id=p.decision.memory_id,
                    content=p.decision.final_content,
                )
                updated = (
                    self.table.search()
                    .where(f"replaces_id = '{p.decision.memory_id}'")
                    .to_list()
                )
                if updated:
                    results.append(self._without_vectors([dict(updated[0])])[0])
            else:
                results.append(
                    self.create_memory(
                        user_id=p.user_id,
                        content=p.decision.final_content,
                        session_id=p.session_id,
                        conversation_id=p.conversation_id,
                        memory_type=p.inferred_type,
                        metadata=p.metadata,
                        scope=p.scope,
                    )
                )
        return results

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
        scope: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        skip_threshold: float = 0.85,
        use_reconciler: bool = True,
        num_threads: int = 4,
    ) -> Memories:
        """Batch upsert: insert, update, or skip memories based on semantic similarity."""
        if extract:
            if not contents:
                raise ValueError("contents is required when extract=True")

            extractor = MemoryExtractor(signature=self._extraction_signature)
            with dspy.context(lm=self._extraction_lm):
                extracted = self._normalize_extracted_memories(
                    cast(list[Any], extractor(messages=contents).memories)
                )

            if use_reconciler:
                reconciler = MemoryReconciler()
                pending = self._reconcile_batch_parallel(
                    extracted=extracted,
                    reconciler=reconciler,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=memory_type,
                    scope=scope,
                    metadata=metadata,
                    skip_threshold=skip_threshold,
                    num_threads=num_threads,
                )
                return self._apply_reconciliations(pending)

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
                    metadata=self._merge_metadata(metadata, item_metadata),
                    scope=scope,
                    similarity_threshold=similarity_threshold,
                    skip_threshold=skip_threshold,
                    use_reconciler=False,
                )
                for content, inferred_type, item_metadata in extracted
            ]
            return stored

        if not contents or len(contents) != 1:
            raise ValueError(
                "verbatim upsert (extract=False) requires exactly one item in contents"
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
            scope=scope,
            similarity_threshold=similarity_threshold,
            skip_threshold=skip_threshold,
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
        scope: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        skip_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> Memory:
        """Insert or update a memory based on semantic similarity."""
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
        fetch_limit = candidate_limit * (10 if scope else 1)
        results = (
            self.table.search(self._embed(content), vector_column_name="vector")
            .where(" AND ".join(filters))
            .limit(fetch_limit)
            .to_list()
        )
        results = self._filter_rows_by_json_fields(
            [dict(row) for row in results], scope=scope
        )[:candidate_limit]

        if results:
            if resolved_type == MemoryType.SEMANTIC.value:
                if use_reconciler:
                    nearest_distance = float(results[0].get("_distance", 1.0))
                    nearest_similarity = 1.0 - nearest_distance
                    if nearest_similarity >= skip_threshold:
                        return self._without_vectors([dict(results[0])])[0]

                    reconciler = MemoryReconciler()
                    existing_list = [
                        {
                            "id": str(c["id"]),
                            "content": str(c["content"]),
                            "type": str(c["memory_type"]),
                        }
                        for c in results
                    ]
                    with dspy.context(lm=self._extraction_lm):
                        decision = reconciler(
                            new_memory_content=content,
                            new_memory_type=str(resolved_type),
                            existing_memories=existing_list,
                        ).reconciled

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
                            .where(f"replaces_id = '{decision.memory_id}'")
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
                            skip_threshold=skip_threshold,
                        )
                        if action == "skip":
                            return self._without_vectors([existing])[0]
                        if action == "update":
                            self.update_memory(
                                memory_id=str(existing["id"]), content=content
                            )
                            updated = (
                                self.table.search()
                                .where(f"replaces_id = '{existing['id']}'")
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

                if cosine_sim >= skip_threshold:
                    return self._without_vectors([existing])[0]

                if cosine_sim >= similarity_threshold:
                    self.update_memory(memory_id=str(existing["id"]), content=content)
                    updated = (
                        self.table.search()
                        .where(f"replaces_id = '{existing['id']}'")
                        .to_list()
                    )
                    return self._without_vectors([dict(updated[0])])[0]

        return self._without_vectors(
            [
                self.create_memory(
                    user_id=user_id,
                    content=content,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=memory_type,
                    metadata=metadata,
                    scope=scope,
                )
            ]
        )[0]

    def delete_memory(self, *, memory_id: str) -> None:
        self.table.update(where=f"id = '{memory_id}'", values={"is_active": False})

    def delete_memories_by_search(
        self,
        *,
        user_id: str,
        query: str,
        session_id: str = "",
        conversation_id: str = "",
        memory_type: MemoryType | str | None = None,
        scope: dict[str, Any] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        limit: int = 5,
    ) -> Memories:
        filters = self._build_filters(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=self._memory_type_value(memory_type),
        )

        fetch_limit = limit * (10 if (scope or metadata_filter) else 1)
        candidates = (
            self.table.search(self._embed(query), vector_column_name="vector")
            .where(" AND ".join(filters))
            .limit(fetch_limit)
            .to_list()
        )

        if not candidates:
            return []

        results = self._filter_rows_by_json_fields(
            [dict(row) for row in candidates],
            scope=scope,
            metadata_filter=metadata_filter,
        )
        results = self._filter_by_relevance(results, min_score=similarity_threshold)
        results = results[:limit]

        if not results:
            return []

        deleted = []
        for row in results:
            memory_id = str(row["id"])
            self.table.update(where=f"id = '{memory_id}'", values={"is_active": False})
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
        scope: dict[str, Any] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.85,
        skip_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> tuple[Memories, Memories]:
        """Extract and process memory operations (create/update/delete) from conversation.

        Returns a tuple of (created_or_updated_memories, deleted_memories).
        """
        if not contents:
            raise ValueError("contents is required")

        if extract:
            extractor = MemoryOperationExtractor()
            with dspy.context(lm=self._extraction_lm):
                operations = extractor(messages=contents).operations
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
                        .limit(10 if (scope or metadata_filter) else 1)
                        .to_list()
                    )
                    exact_results = self._filter_rows_by_json_fields(
                        [dict(row) for row in exact_results],
                        scope=scope,
                        metadata_filter=metadata_filter,
                    )
                    if exact_results:
                        for row in exact_results[:1]:
                            memory_id = str(row["id"])
                            self.table.update(
                                where=f"id = '{memory_id}'",
                                values={"is_active": False},
                            )
                            deleted.append(self._to_memory(dict(row)))
                        continue

                removed = self.delete_memories_by_search(
                    user_id=user_id,
                    query=search_query,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    memory_type=self._memory_type_value(op.memory_type),
                    scope=scope,
                    metadata_filter=metadata_filter,
                    similarity_threshold=similarity_threshold,
                    limit=1,
                )
                deleted.extend(removed)
                continue

            if action in ("create", "update") and op.content:
                created_or_updated.append(
                    self.upsert_memory(
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        content=op.content,
                        memory_type=(
                            memory_type_from_string(op.memory_type)
                            if op.memory_type
                            else MemoryType.SEMANTIC
                        ),
                        metadata=metadata,
                        scope=scope,
                        similarity_threshold=similarity_threshold,
                        skip_threshold=skip_threshold,
                        use_reconciler=use_reconciler,
                    )
                )

        return created_or_updated, deleted
