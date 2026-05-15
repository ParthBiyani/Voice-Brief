"""Qdrant access.

Three logical collections, deliberately kept separate rather than namespaced into one:

  items          — the daily crawl. High churn, pruned on a rolling window.
  episode_memory — segment vectors for "which paper did you mention last week?".
                   Append-only, never pruned; this is the product's memory.
  documents      — Mode 2 uploads, one namespace per document via a payload filter.

Separate collections mean the pruning policy for one cannot corrupt another, and the
episode memory survives any amount of ingest churn.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)

ITEMS = "vb_items"
EPISODE_MEMORY = "vb_episode_memory"
DOCUMENTS = "vb_documents"

ITEM_RETENTION_DAYS = 30


@dataclass(slots=True)
class Neighbour:
    id: str
    score: float
    payload: dict


class VectorStore:
    def __init__(self, url: str | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self.dim = dim or settings.embed_dim
        self.client = QdrantClient(url=url or settings.qdrant_url, timeout=30)

    # ── schema ────────────────────────────────────────────────────────────────
    def ensure_collections(self) -> None:
        """Idempotent. Safe to call on every boot."""
        for name in (ITEMS, EPISODE_MEMORY, DOCUMENTS):
            self._ensure(name)

        # Payload indexes matter: without them Qdrant filters by scanning, which
        # turns the per-document Mode 2 search into a full-collection walk.
        self._ensure_index(ITEMS, "published_at", qm.PayloadSchemaType.FLOAT)
        self._ensure_index(ITEMS, "source_slug", qm.PayloadSchemaType.KEYWORD)
        self._ensure_index(EPISODE_MEMORY, "episode_id", qm.PayloadSchemaType.KEYWORD)
        self._ensure_index(EPISODE_MEMORY, "user_id", qm.PayloadSchemaType.KEYWORD)
        self._ensure_index(DOCUMENTS, "document_id", qm.PayloadSchemaType.KEYWORD)

    def _ensure(self, name: str) -> None:
        if self.client.collection_exists(name):
            existing = self.client.get_collection(name)
            size = existing.config.params.vectors.size
            if size != self.dim:
                raise ValueError(
                    f"Qdrant collection {name!r} has {size}-d vectors but the configured "
                    f"embedding model produces {self.dim}. Changing embedding models "
                    f"requires re-indexing, not a config edit."
                )
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        log.info("qdrant.collection_created", name=name, dim=self.dim)

    def _ensure_index(self, collection: str, field: str, schema) -> None:
        try:
            self.client.create_payload_index(
                collection_name=collection, field_name=field, field_schema=schema, wait=True
            )
        except Exception as exc:  # noqa: BLE001 — index already exists is not an error
            log.debug("qdrant.index_exists", collection=collection, field=field, error=str(exc))

    # ── writes ────────────────────────────────────────────────────────────────
    def upsert_items(
        self, ids: Sequence[uuid.UUID], vectors: np.ndarray, payloads: Sequence[dict]
    ) -> None:
        if len(ids) == 0:
            return
        self.client.upsert(
            collection_name=ITEMS,
            points=qm.Batch(
                ids=[str(i) for i in ids],
                vectors=vectors.tolist(),
                payloads=list(payloads),
            ),
            wait=True,
        )

    def upsert_segments(
        self, ids: Sequence[uuid.UUID], vectors: np.ndarray, payloads: Sequence[dict]
    ) -> None:
        if len(ids) == 0:
            return
        self.client.upsert(
            collection_name=EPISODE_MEMORY,
            points=qm.Batch(
                ids=[str(i) for i in ids], vectors=vectors.tolist(), payloads=list(payloads)
            ),
            wait=True,
        )

    def upsert_chunks(
        self, ids: Sequence[str], vectors: np.ndarray, payloads: Sequence[dict]
    ) -> None:
        if len(ids) == 0:
            return
        self.client.upsert(
            collection_name=DOCUMENTS,
            points=qm.Batch(ids=list(ids), vectors=vectors.tolist(), payloads=list(payloads)),
            wait=True,
        )

    # ── reads ─────────────────────────────────────────────────────────────────
    def search_items(
        self, vector: np.ndarray, *, limit: int = 10, score_threshold: float | None = None
    ) -> list[Neighbour]:
        hits = self.client.query_points(
            collection_name=ITEMS,
            query=vector.tolist(),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
        return [Neighbour(id=str(h.id), score=h.score, payload=h.payload or {}) for h in hits]

    def search_memory(
        self, vector: np.ndarray, *, user_id: uuid.UUID | None = None, limit: int = 10
    ) -> list[Neighbour]:
        flt = None
        if user_id:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=str(user_id)))]
            )
        hits = self.client.query_points(
            collection_name=EPISODE_MEMORY,
            query=vector.tolist(),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
        return [Neighbour(id=str(h.id), score=h.score, payload=h.payload or {}) for h in hits]

    def search_document(
        self, vector: np.ndarray, *, document_ids: Sequence[str], limit: int = 8
    ) -> list[Neighbour]:
        flt = qm.Filter(
            must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=list(document_ids)))]
        )
        hits = self.client.query_points(
            collection_name=DOCUMENTS,
            query=vector.tolist(),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
        return [Neighbour(id=str(h.id), score=h.score, payload=h.payload or {}) for h in hits]

    def fetch_vectors(self, ids: Iterable[uuid.UUID]) -> dict[str, np.ndarray]:
        ids = [str(i) for i in ids]
        if not ids:
            return {}
        records = self.client.retrieve(
            collection_name=ITEMS, ids=ids, with_vectors=True, with_payload=False
        )
        return {str(r.id): np.array(r.vector, dtype=np.float32) for r in records}

    # ── maintenance ───────────────────────────────────────────────────────────
    def prune_items(self, *, older_than_days: int = ITEM_RETENTION_DAYS) -> None:
        """Drop stale item vectors.

        Only the `items` collection is pruned. Episode memory is the product feature
        that answers "what did you mention last week", so deleting from it would be
        deleting the product.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).timestamp()
        self.client.delete(
            collection_name=ITEMS,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="published_at", range=qm.Range(lt=cutoff))]
                )
            ),
            wait=True,
        )
        log.info("qdrant.pruned", collection=ITEMS, older_than_days=older_than_days)

    def count(self, collection: str) -> int:
        return self.client.count(collection_name=collection, exact=True).count


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
