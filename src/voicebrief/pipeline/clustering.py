"""Topic clustering.

Dedup answers "is this the same item?". Clustering answers "is this the same *story*?"
— an arXiv paper, the framework release that implements it, and the blog post
explaining it are three distinct items that belong in one segment.

HDBSCAN rather than k-means, for one decisive reason: **it does not require k**. The
number of genuine stories in a day's crawl varies from about 20 to about 80, and any
fixed k either shreds real stories or fuses unrelated ones. HDBSCAN also labels
outliers as noise (-1) instead of forcing every item into a cluster, which matters
because most of a 300-item crawl genuinely is one-off noise.

Cosine distance on L2-normalized vectors, via the Euclidean metric: for unit vectors,
euclidean² = 2(1 − cosine), so Euclidean HDBSCAN is order-equivalent to cosine and
avoids sklearn's slower precomputed-matrix path.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from voicebrief.logging import get_logger

log = get_logger(__name__)

# Two items are enough to be a story: a release plus the discussion about it is the
# single most common shape in the corpus. Requiring three loses those entirely.
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1

# "leaf", not the sklearn default "eom".
#
# Measured on a real 300-item crawl: excess-of-mass selection collapsed 207 of 300
# items into a single cluster as soon as min_cluster_size reached 3. That is the
# known failure mode of EOM on text embeddings — everything in an AI news corpus is
# mildly similar to everything else (measured mean pairwise cosine 0.61), so the
# density landscape has one broad basin and EOM happily selects it.
#
# Leaf selection takes the finest-grained clusters in the condensed tree instead, and
# stayed stable across every parameter setting tried: 55 clusters, largest 9. A brief
# built on one 207-item "story" would be worthless, so stability matters more here
# than the slightly higher noise rate leaf produces.
CLUSTER_SELECTION = "leaf"

# Below this, HDBSCAN has too little to work with and returns all-noise; fall back to
# treating every item as its own singleton rather than silently emptying the brief.
MIN_ITEMS_FOR_CLUSTERING = 10


def _is_degenerate(vectors: np.ndarray, *, tolerance: float = 1e-6) -> bool:
    """True when every vector is effectively the same point."""
    if len(vectors) < 2:
        return True
    return bool(np.max(np.abs(vectors - vectors[0])) < tolerance)


@dataclass(slots=True)
class Cluster:
    label: int
    member_ids: list[uuid.UUID] = field(default_factory=list)
    centroid_id: uuid.UUID | None = None
    topics: list[str] = field(default_factory=list)
    cohesion: float = 0.0

    @property
    def size(self) -> int:
        return len(self.member_ids)


def cluster_items(
    item_ids: list[uuid.UUID],
    vectors: np.ndarray,
    *,
    topics_by_id: dict[uuid.UUID, list[str]] | None = None,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int = MIN_SAMPLES,
    keep_noise_as_singletons: bool = True,
) -> list[Cluster]:
    """Group items into stories.

    Returns clusters sorted largest-first. When `keep_noise_as_singletons` is set,
    HDBSCAN's noise points become one-item clusters — a genuinely unique story is not
    less newsworthy for being unique, and dropping them would quietly bias the brief
    toward whatever gets written about most.
    """
    if len(item_ids) != len(vectors):
        raise ValueError(f"{len(item_ids)} ids but {len(vectors)} vectors")
    if not item_ids:
        return []

    topics_by_id = topics_by_id or {}

    if len(item_ids) < MIN_ITEMS_FOR_CLUSTERING:
        labels = np.arange(len(item_ids))
        log.info("clustering.too_few_items", count=len(item_ids), fallback="singletons")
    elif _is_degenerate(vectors):
        # HDBSCAN labels a zero-variance set entirely as noise: with all mutual
        # reachability distances equal there is no density structure to find. The
        # singleton fallback would then split one story into N identical segments.
        # Dedup normally removes this case upstream, but depending on stage ordering
        # for correctness is a footgun, so collapse it here too.
        labels = np.zeros(len(item_ids), dtype=int)
        log.warning("clustering.degenerate_input", count=len(item_ids))
    else:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # order-equivalent to cosine on unit vectors
            cluster_selection_method=CLUSTER_SELECTION,
            # sklearn's default is copy=False, which mutates the caller's array in
            # place. The same vectors are reused for dedup and for Qdrant payloads,
            # so silent mutation here would corrupt both.
            copy=True,
        )
        labels = model.fit_predict(vectors)

    clusters: list[Cluster] = []
    next_synthetic = int(labels.max()) + 1 if len(labels) else 0

    grouped: dict[int, list[int]] = {}
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label == -1:
            if not keep_noise_as_singletons:
                continue
            label = next_synthetic
            next_synthetic += 1
        grouped.setdefault(label, []).append(index)

    for label, indices in grouped.items():
        member_vectors = vectors[indices]
        centroid = member_vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # The centroid is an average and therefore not a real item; report the member
        # closest to it, because the script generator needs an actual document to cite.
        similarities = member_vectors @ centroid
        best = int(np.argmax(similarities))

        members = [item_ids[i] for i in indices]
        clusters.append(
            Cluster(
                label=label,
                member_ids=members,
                centroid_id=item_ids[indices[best]],
                topics=_dominant_topics(members, topics_by_id),
                cohesion=float(similarities.mean()),
            )
        )

    clusters.sort(key=lambda c: (c.size, c.cohesion), reverse=True)

    noise = int((labels == -1).sum()) if len(labels) else 0
    log.info(
        "clustering.done",
        items=len(item_ids),
        clusters=len(clusters),
        multi_item=sum(1 for c in clusters if c.size > 1),
        noise=noise,
    )
    return clusters


def _dominant_topics(
    member_ids: list[uuid.UUID], topics_by_id: dict[uuid.UUID, list[str]], limit: int = 4
) -> list[str]:
    counter: Counter[str] = Counter()
    for member_id in member_ids:
        counter.update(t.lower() for t in topics_by_id.get(member_id, []))
    return [topic for topic, _ in counter.most_common(limit)]


def labels_from_clusters(clusters: list[Cluster]) -> dict[uuid.UUID, int]:
    """Flatten to an id → label map, which is the shape the ARI metric wants."""
    return {member_id: cluster.label for cluster in clusters for member_id in cluster.member_ids}
