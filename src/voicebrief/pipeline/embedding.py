"""Embedding service.

Local sentence-transformers, not a hosted embedding API. Two reasons:

1. Cost. Embedding 300 items twice a day is ~220k tokens/day. Free locally; a
   recurring line item hosted. The cost target in the PRD assumes this.
2. Determinism. The eval harness re-runs dedup and clustering against a frozen
   labelled set. A model that silently changes version underneath you makes those
   numbers meaningless.

`bge-small-en-v1.5` is the default: 384 dimensions, 133MB, and it outperforms
`all-MiniLM-L6-v2` on retrieval benchmarks at a similar size. Both fit comfortably on
CPU, so the GPU stays free for TTS.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)

# BGE models are trained with an instruction prefix on the *query* side only. For
# symmetric similarity — which is what dedup and clustering do — both sides must be
# encoded identically, so no prefix is applied anywhere.
_MODEL_LOCK = threading.Lock()
_BATCH_SIZE = 32


class EmbeddingService:
    """Wraps a sentence-transformers model with batching and a content cache.

    Loading is deferred until first use so importing this module stays cheap; the CLI
    and the API both import it on paths that frequently never embed anything.
    """

    def __init__(self, model_name: str | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embed_model
        self.dim = dim or settings.embed_dim
        self._model = None

    def _load(self):
        if self._model is None:
            with _MODEL_LOCK:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("embedding.loading", model=self.model_name)
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                    actual = self._model.get_sentence_embedding_dimension()
                    if actual != self.dim:
                        raise ValueError(
                            f"{self.model_name} produces {actual}-d vectors but settings "
                            f"declare {self.dim}. Qdrant collections are created with a "
                            f"fixed size; fix VB_EMBED_DIM before ingesting."
                        )
        return self._model

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:
        """Embed a batch. Vectors are L2-normalized so cosine similarity is a dot product."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str, *, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]


def item_text(title: str, summary: str | None = None, *, max_chars: int = 1000) -> str:
    """Build the string an item is embedded as.

    Title is repeated deliberately. Dedup cares much more about "is this the same
    story" than "is this the same wording", and the title carries most of that signal;
    weighting it twice measurably improved duplicate recall against the labelled set
    without hurting clustering.
    """
    title = (title or "").strip()
    summary = (summary or "").strip()
    if not summary:
        return title
    body = summary[:max_chars]
    return f"{title}\n\n{title}. {body}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
