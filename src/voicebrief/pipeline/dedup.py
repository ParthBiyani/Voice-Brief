"""Near-duplicate detection.

The same story reaches us many times over: an arXiv paper, the Hacker News thread
about it, the newsletter that covered it, and the lab's own blog post. Reporting all
four as separate segments is the fastest way to make a brief feel like a feed reader.

Two passes, cheapest first:

1. **Exact.** URL and title fingerprints. Catches syndication and cross-posting for
   free, no vectors involved.
2. **Semantic.** Cosine similarity over embeddings above a threshold (PRD: 0.92).

Duplicates are grouped, not deleted. Each group elects a canonical item — highest
source trust, then most engagement, then earliest published — and the rest point at it
via `duplicate_of_id`. Keeping them matters: "covered by arXiv, HN and Import AI" is
itself a signal that a story is important, and the script generator uses it.

Union-Find rather than pairwise clustering because duplication is transitive in
practice (A≈B, B≈C ⟹ all one story) and the groups must be stable regardless of the
order items arrive in — the eval harness depends on that determinism.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from voicebrief.logging import get_logger

log = get_logger(__name__)

DEFAULT_THRESHOLD = 0.92

# Tracking parameters carry no meaning for identity.
_TRACKING_PARAMS = re.compile(
    r"[?&](utm_[a-z]+|ref|referrer|source|fbclid|gclid|mc_cid|mc_eid|__s)=[^&]*",
    re.IGNORECASE,
)
_TITLE_NOISE = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def canonical_url(url: str) -> str:
    """Strip tracking, trailing slashes, and scheme/host casing."""
    if not url:
        return ""
    url = _TRACKING_PARAMS.sub("", url)
    url = url.split("#", 1)[0].rstrip("?&").rstrip("/")
    if "://" in url:
        scheme, rest = url.split("://", 1)
        host, _, path = rest.partition("/")
        host = host.lower().removeprefix("www.")
        url = f"{scheme.lower()}://{host}" + (f"/{path}" if path else "")
    return url


def normalized_title(title: str) -> str:
    """Lowercase, punctuation-free, whitespace-collapsed.

    Enough to match "LangGraph 0.4: Durable Execution" with "LangGraph 0.4 - durable
    execution" without pulling in a stemmer.
    """
    return _WS.sub(" ", _TITLE_NOISE.sub(" ", (title or "").lower())).strip()


def url_fingerprint(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:32]


def title_fingerprint(title: str) -> str:
    return hashlib.sha256(normalized_title(title).encode()).hexdigest()[:32]


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Union by smaller index so group membership is order-independent.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self._parent[hi] = lo

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for node in list(self._parent):
            out[self.find(node)].append(node)
        return out


@dataclass(slots=True)
class DedupCandidate:
    """The subset of an item dedup needs. Decoupled from the ORM so the eval harness
    can run the exact same code against labelled fixtures."""

    id: uuid.UUID
    title: str
    url: str
    trust_weight: float = 0.5
    engagement: float = 0.0
    published_ts: float = 0.0


@dataclass(slots=True)
class DuplicateGroup:
    canonical_id: uuid.UUID
    member_ids: list[uuid.UUID] = field(default_factory=list)
    reason: str = "semantic"

    @property
    def size(self) -> int:
        return len(self.member_ids) + 1


def _elect_canonical(members: list[DedupCandidate]) -> DedupCandidate:
    """Highest trust, then most engagement, then earliest published, then id.

    The final id tiebreak exists so the choice is deterministic even for two items
    that are identical on every real signal — the eval harness compares group
    membership across runs and would otherwise show phantom diffs.
    """
    return max(
        members,
        key=lambda c: (c.trust_weight, c.engagement, -c.published_ts, str(c.id)),
    )


def find_duplicates(
    candidates: list[DedupCandidate],
    vectors: np.ndarray | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[DuplicateGroup]:
    """Group near-duplicates.

    `vectors[i]` must correspond to `candidates[i]` and be L2-normalized. Passing
    None runs exact matching only, which is what the cost ablation compares against.
    """
    if len(candidates) < 2:
        return []

    uf = _UnionFind()
    reasons: dict[int, str] = {}

    # Pass 1 — exact. Free, and catches the bulk of real-world syndication.
    by_url: dict[str, int] = {}
    by_title: dict[str, int] = {}
    for i, c in enumerate(candidates):
        uf.find(i)
        ufp = url_fingerprint(c.url)
        if ufp and ufp in by_url:
            uf.union(by_url[ufp], i)
            reasons[i] = "url"
        else:
            by_url[ufp] = i

        tfp = title_fingerprint(c.title)
        if tfp and tfp in by_title:
            uf.union(by_title[tfp], i)
            reasons.setdefault(i, "title")
        else:
            by_title[tfp] = i

    # Pass 2 — semantic.
    if vectors is not None and len(vectors) == len(candidates):
        similarity = vectors @ vectors.T
        # Upper triangle only; the matrix is symmetric and the diagonal is self-match.
        rows, cols = np.triu_indices(len(candidates), k=1)
        for i, j in zip(rows[similarity[rows, cols] >= threshold],
                        cols[similarity[rows, cols] >= threshold], strict=True):
            uf.union(int(i), int(j))
            reasons.setdefault(int(j), "semantic")

    groups: list[DuplicateGroup] = []
    for _, indices in uf.groups().items():
        if len(indices) < 2:
            continue
        members = [candidates[i] for i in indices]
        canonical = _elect_canonical(members)
        others = [m.id for m in members if m.id != canonical.id]
        group_reason = next(
            (reasons[i] for i in sorted(indices) if i in reasons), "semantic"
        )
        groups.append(
            DuplicateGroup(canonical_id=canonical.id, member_ids=others, reason=group_reason)
        )

    log.info(
        "dedup.done",
        candidates=len(candidates),
        groups=len(groups),
        collapsed=sum(len(g.member_ids) for g in groups),
        threshold=threshold,
    )
    return groups


def pairs_from_groups(groups: list[DuplicateGroup]) -> set[frozenset[uuid.UUID]]:
    """Expand groups into the unordered pairs the evaluation metrics operate on."""
    pairs: set[frozenset[uuid.UUID]] = set()
    for group in groups:
        members = [group.canonical_id, *group.member_ids]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(frozenset((members[i], members[j])))
    return pairs
