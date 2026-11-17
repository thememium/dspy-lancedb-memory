"""Reconciliation-driven upsert pipeline."""

from __future__ import annotations

from typing import Any, cast

import dspy

from ..extraction import MemoryExtractor, MemoryReconciler
from ..models import (
    Memories,
    Memory,
    MemoryType,
    PendingReconciliation,
    ReconciledMemory,
    ScopeLike,
    memory_type_from_string,
    memory_type_value,
)
from .filters import build_filters, filter_rows_by_json_fields
from .rows import normalize_extracted_memories, without_vectors
from .serialization import merge_metadata, scope_dict
from .similarity import distance_to_similarity, semantic_match_action


class SingleMemoryReconciler:
    """Callable wrapper for dspy.Parallel — reconciles one memory."""

    def __init__(
        self,
        store: UpsertMixin,
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


class UpsertMixin:
    """Similarity-aware insertion and update of memories."""

    _UPSERT_CANDIDATE_LIMIT = 10

    table: Any
    _extraction_lm: dspy.LM | None
    _extraction_signature: type[dspy.Signature]

    _embed: Any
    _select_all: Any
    create_memory: Any
    update_memory: Any

    def _reconcile_one(
        self,
        content: str,
        inferred_type: MemoryType | str,
        memory_type: MemoryType | str | None,
        user_id: str,
        session_id: str,
        conversation_id: str,
        scope: ScopeLike | None,
        metadata: dict[str, Any],
        skip_threshold: float,
        reconciler: MemoryReconciler,
    ) -> PendingReconciliation:
        memory_type_str = (
            memory_type_from_string(memory_type)
            if memory_type is not None
            else inferred_type
        )
        memory_type_str_value = memory_type_value(memory_type_str)

        filters = build_filters(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=(
                memory_type_str_value
                if memory_type_str_value == MemoryType.SEMANTIC.value
                else None
            ),
        )
        candidate_limit = self._UPSERT_CANDIDATE_LIMIT
        fetch_limit = candidate_limit * (10 if scope else 1)
        candidates = (
            self._select_all(
                self.table.search(self._embed(content), vector_column_name="vector"),
                "_distance",
            )
            .where(" AND ".join(filters))
            .limit(fetch_limit)
            .to_list()
        )
        candidates = filter_rows_by_json_fields(
            [dict(row) for row in candidates], scope=scope
        )[:candidate_limit]

        if candidates:
            nearest_distance = float(candidates[0].get("_distance", 1.0))
            nearest_similarity = distance_to_similarity(nearest_distance)
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
                    scope=scope_dict(scope),
                    metadata=metadata,
                    existing_row=without_vectors([dict(candidates[0])])[0],
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
            scope=scope_dict(scope),
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
        scope: ScopeLike | None,
        metadata: dict[str, Any] | None,
        skip_threshold: float,
        num_threads: int,
    ) -> list[PendingReconciliation]:
        exec_pairs: list[tuple[Any, dspy.Example]] = []
        for content, inferred_type, item_metadata in extracted:
            module = SingleMemoryReconciler(
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
                scope=scope_dict(scope),
                metadata=merge_metadata(metadata, item_metadata),
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
        for item in pending:
            if item.decision.action == "keep":
                if hasattr(item, "existing_row") and item.existing_row:
                    results.append(item.existing_row)
                else:
                    kept = (
                        self.table.search()
                        .where(f"id = '{item.decision.memory_id}'")
                        .to_list()
                    )
                    if kept:
                        results.append(without_vectors([dict(kept[0])])[0])
            elif item.decision.action == "update":
                self.update_memory(
                    memory_id=item.decision.memory_id,
                    content=item.decision.final_content,
                )
                updated = (
                    self.table.search()
                    .where(f"replaces_id = '{item.decision.memory_id}'")
                    .to_list()
                )
                if updated:
                    results.append(without_vectors([dict(updated[0])])[0])
            else:
                results.append(
                    self.create_memory(
                        user_id=item.user_id,
                        content=item.decision.final_content,
                        session_id=item.session_id,
                        conversation_id=item.conversation_id,
                        memory_type=item.inferred_type,
                        metadata=item.metadata,
                        scope=item.scope,
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
        scope: ScopeLike | None = None,
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
                extracted = normalize_extracted_memories(
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
                    metadata=merge_metadata(metadata, item_metadata),
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
        scope: ScopeLike | None = None,
        similarity_threshold: float = 0.85,
        skip_threshold: float = 0.85,
        use_reconciler: bool = True,
    ) -> Memory:
        """Insert or update a memory based on semantic similarity."""
        resolved_type = memory_type_value(memory_type)
        candidate_limit = self._UPSERT_CANDIDATE_LIMIT

        filters = build_filters(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_type=(
                resolved_type if resolved_type == MemoryType.SEMANTIC.value else None
            ),
        )
        fetch_limit = candidate_limit * (10 if scope else 1)
        results = (
            self._select_all(
                self.table.search(self._embed(content), vector_column_name="vector"),
                "_distance",
            )
            .where(" AND ".join(filters))
            .limit(fetch_limit)
            .to_list()
        )
        results = filter_rows_by_json_fields(
            [dict(row) for row in results], scope=scope
        )[:candidate_limit]

        if results:
            if resolved_type == MemoryType.SEMANTIC.value:
                if use_reconciler:
                    nearest_distance = float(results[0].get("_distance", 1.0))
                    nearest_similarity = distance_to_similarity(nearest_distance)
                    if nearest_similarity >= skip_threshold:
                        return without_vectors([dict(results[0])])[0]

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
                            return without_vectors([dict(kept[0])])[0]
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
                            return without_vectors([dict(updated[0])])[0]
                else:
                    for candidate in results:
                        existing = dict(candidate)
                        action = semantic_match_action(
                            new_content=content,
                            existing_content=str(existing["content"]),
                            distance=float(existing.get("_distance", 1.0)),
                            similarity_threshold=similarity_threshold,
                            skip_threshold=skip_threshold,
                        )
                        if action == "skip":
                            return without_vectors([existing])[0]
                        if action == "update":
                            self.update_memory(
                                memory_id=str(existing["id"]), content=content
                            )
                            updated = (
                                self.table.search()
                                .where(f"replaces_id = '{existing['id']}'")
                                .to_list()
                            )
                            return without_vectors([dict(updated[0])])[0]
            else:
                for candidate in results:
                    existing = dict(candidate)
                    if existing["content"] == content:
                        return without_vectors([existing])[0]

                existing = dict(results[0])
                distance = float(existing.get("_distance", 1.0))
                cosine_sim = distance_to_similarity(distance)

                if cosine_sim >= skip_threshold:
                    return without_vectors([existing])[0]

                if cosine_sim >= similarity_threshold:
                    self.update_memory(memory_id=str(existing["id"]), content=content)
                    updated = (
                        self.table.search()
                        .where(f"replaces_id = '{existing['id']}'")
                        .to_list()
                    )
                    return without_vectors([dict(updated[0])])[0]

        return without_vectors(
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
