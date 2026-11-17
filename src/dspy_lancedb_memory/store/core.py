"""The LanceDB-backed memory store."""

from __future__ import annotations

import atexit
import logging
import weakref

import dspy
import lancedb
from lancedb.rerankers import Reranker

from ..extraction import ExtractMemory
from .embeddings import EmbeddingMixin, build_embedder
from .process import ProcessMixin
from .read import ReadMixin
from .search import SearchMixin
from .table import TableMixin
from .upsert import UpsertMixin
from .writes import WriteMixin

logger = logging.getLogger("dspy_lancedb_memory")


class LanceDSPyMemoryStore(
    EmbeddingMixin,
    TableMixin,
    WriteMixin,
    ReadMixin,
    SearchMixin,
    UpsertMixin,
    ProcessMixin,
):
    """Persistent, append-only memory store backed by LanceDB."""

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
        rte_durable_write: bool = False,
    ):
        self.db = lancedb.connect(uri)
        self.table_uri = uri
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.rerank_limit_multiplier = max(rerank_limit_multiplier, 1)
        self.rte_durable_write = rte_durable_write
        self._extraction_lm = extraction_lm

        self.embedder = build_embedder(embedding_lm)
        self.reranker = reranker
        self._extraction_signature = signature or ExtractMemory

        if self.embedding_dim is None:
            self.embedding_dim = self._infer_embedding_dim()

        self.table = self._get_or_create_table()
        self._register_flush_on_exit()

    def _register_flush_on_exit(self) -> None:
        ref = weakref.ref(self)

        def _flush_on_exit() -> None:
            store = ref()
            if store is None:
                return
            try:
                store.flush()
            except Exception:
                logger.debug("RTE flush on exit failed", exc_info=True)

        atexit.register(_flush_on_exit)
