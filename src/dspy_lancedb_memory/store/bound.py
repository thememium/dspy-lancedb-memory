"""A memory store with user/session/conversation/scope pre-bound."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..models import Memories, Memory, ScopeLike
from .serialization import scope_dict

if TYPE_CHECKING:
    from .core import LanceDSPyMemoryStore


class BoundMemoryStore:
    """A convenience wrapper with user/session/conversation/scope pre-bound."""

    def __init__(
        self,
        store: LanceDSPyMemoryStore,
        *,
        user_id: str,
        session_id: str = "",
        conversation_id: str = "",
        scope: ScopeLike | None = None,
    ) -> None:
        self.store = store
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.scope = scope_dict(scope)

    def _scope(self, scope: ScopeLike | None = None) -> dict[str, Any]:
        merged = dict(self.scope)
        merged.update(scope_dict(scope))
        return merged

    def create_memory(
        self,
        *,
        content: str,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memory:
        return self.store.create_memory(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            content=content,
            scope=self._scope(scope),
            **kwargs,
        )

    def create_memories(
        self,
        *,
        contents: list[dict[str, str]] | None = None,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memories:
        return self.store.create_memories(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            contents=contents,
            scope=self._scope(scope),
            **kwargs,
        )

    def search_memories(
        self,
        *,
        query: str,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memories:
        return self.store.search_memories(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            query=query,
            scope=self._scope(scope),
            **kwargs,
        )

    def upsert_memory(
        self,
        *,
        content: str,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memory:
        return self.store.upsert_memory(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            content=content,
            scope=self._scope(scope),
            **kwargs,
        )

    def upsert_memories(
        self,
        *,
        contents: list[dict[str, str]] | None = None,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memories:
        return self.store.upsert_memories(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            contents=contents,
            scope=self._scope(scope),
            **kwargs,
        )

    def list_memories(
        self,
        *,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memories:
        return self.store.list_memories(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            scope=self._scope(scope),
            **kwargs,
        )

    def process_memories(
        self,
        *,
        contents: list[dict[str, str]] | None = None,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> tuple[Memories, Memories]:
        return self.store.process_memories(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            contents=contents,
            scope=self._scope(scope),
            **kwargs,
        )

    def delete_memories_by_search(
        self,
        *,
        query: str,
        scope: ScopeLike | None = None,
        **kwargs: Any,
    ) -> Memories:
        return self.store.delete_memories_by_search(
            user_id=self.user_id,
            session_id=kwargs.pop("session_id", self.session_id),
            conversation_id=kwargs.pop("conversation_id", self.conversation_id),
            query=query,
            scope=self._scope(scope),
            **kwargs,
        )
