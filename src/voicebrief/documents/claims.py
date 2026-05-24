"""Claim extraction and the cross-document graph.

This is what makes Mode 2 a contrast engine rather than a summarizer. Summarizing
five papers gives you five summaries; the useful output is *where they disagree*, and
that requires comparing assertions, not prose.

Two stages:

1. **Extract** — pull standalone claims from each document, one call per document.
2. **Relate** — compare claims that are semantically near each other and label the
   relationship: agrees, contradicts, extends, prerequisite.

Only near-neighbour pairs are compared. Comparing all pairs across five documents
with twelve claims each is 1,770 comparisons; embedding-gated, it is a few dozen. The
gate is what makes contrast affordable, and it costs almost nothing in recall — two
claims that disagree are, by construction, about the same thing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from voicebrief.llm.base import Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger
from voicebrief.pipeline.embedding import get_embedding_service

log = get_logger(__name__)

# Claims below this similarity are not about the same thing, so they cannot
# meaningfully agree or disagree.
RELATION_THRESHOLD = 0.55
MAX_PAIRS_PER_RUN = 60
MAX_CLAIMS_PER_DOCUMENT = 12

RELATIONS = ("agrees", "contradicts", "extends", "prerequisite", "unrelated")


EXTRACT_SYSTEM = """\
You extract the checkable claims a document actually makes.

A claim is a standalone assertion that could be agreed with or disputed by another
document. "Transformers scale better than LSTMs on long sequences" is a claim.
"This paper is organized as follows" is not.

Rules:
- Each claim must stand alone, without needing the surrounding text to make sense.
- Use the document's own terms. Do not generalise or soften.
- Prefer claims with a specific subject: a method, a result, a limitation.
- Skip background that the document reports rather than asserts.
- Confidence reflects how strongly the document commits: a measured result is high,
  a suggestion in future work is low.

Return JSON only.\
"""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "section": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["text", "section", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


RELATE_SYSTEM = """\
You compare pairs of claims from different documents and label the relationship.

- `contradicts` — both cannot be true as stated. This is the label that matters most;
  do not use it for claims that merely differ in emphasis or scope.
- `agrees` — they assert substantially the same thing.
- `extends` — the second builds on the first with additional specificity.
- `prerequisite` — the first must hold for the second to make sense.
- `unrelated` — they are about different things despite surface similarity.

`unrelated` is the correct answer more often than it feels like it should be. Two
claims that share vocabulary are not thereby in dialogue. A fabricated disagreement
is worse than a missed one, because the episode will state it as fact.

Return JSON only.\
"""

RELATE_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_index": {"type": "integer"},
                    "relation": {"type": "string", "enum": list(RELATIONS)},
                    "rationale": {"type": "string"},
                },
                "required": ["pair_index", "relation", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class Claim:
    id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    text: str
    section: str = ""
    confidence: float = 0.5


@dataclass(slots=True)
class Relation:
    source: Claim
    target: Claim
    relation: str
    rationale: str = ""
    similarity: float = 0.0


@dataclass(slots=True)
class ClaimGraph:
    claims: list[Claim] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    @property
    def contradictions(self) -> list[Relation]:
        return [r for r in self.relations if r.relation == "contradicts"]

    @property
    def agreements(self) -> list[Relation]:
        return [r for r in self.relations if r.relation == "agrees"]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for relation in self.relations:
            counts[relation.relation] = counts.get(relation.relation, 0) + 1
        return {
            "claims": len(self.claims),
            "documents": len({c.document_id for c in self.claims}),
            "relations": counts,
        }


def extract_claims(
    client: LLMClient,
    *,
    document_id: uuid.UUID,
    document_title: str,
    chunks: list,
    max_claims: int = MAX_CLAIMS_PER_DOCUMENT,
) -> list[Claim]:
    """One call per document over its most substantial chunks."""
    if not chunks:
        return []

    # The longest chunks carry the argument; the short ones are usually headings,
    # captions and boilerplate.
    ranked = sorted(chunks, key=lambda c: -len(c.text))[:8]
    body = "\n\n".join(f"[{c.section}]\n{c.text[:1500]}" for c in ranked)

    try:
        completion = client.complete(
            prompt=(
                f"DOCUMENT: {document_title}\n\n{body}\n\n"
                f"Extract at most {max_claims} claims this document makes."
            ),
            stage=Stage.claim_extraction,
            tier=Tier.utility,
            system=EXTRACT_SYSTEM,
            schema=EXTRACT_SCHEMA,
            max_tokens=2000,
            cache_system=True,
        )
    except Exception as exc:  # noqa: BLE001 — one document, not the episode
        log.warning("claims.extract_failed", document=document_title, error=str(exc))
        return []

    claims = []
    for entry in (completion.parsed or {}).get("claims", [])[:max_claims]:
        text = str(entry.get("text", "")).strip()
        if len(text) < 20:
            continue
        claims.append(
            Claim(
                id=uuid.uuid4(),
                document_id=document_id,
                document_title=document_title,
                text=text,
                section=str(entry.get("section", ""))[:200],
                confidence=float(entry.get("confidence", 0.5)),
            )
        )
    log.info("claims.extracted", document=document_title, count=len(claims))
    return claims


def candidate_pairs(
    claims: list[Claim], *, threshold: float = RELATION_THRESHOLD, limit: int = MAX_PAIRS_PER_RUN
) -> list[tuple[Claim, Claim, float]]:
    """Cross-document claim pairs that are close enough to be in dialogue.

    Same-document pairs are excluded: a document agreeing with itself is not a
    finding, and it would flood the graph.
    """
    if len(claims) < 2:
        return []

    vectors = get_embedding_service().encode([c.text for c in claims])
    pairs: list[tuple[Claim, Claim, float]] = []

    for i, j in combinations(range(len(claims)), 2):
        if claims[i].document_id == claims[j].document_id:
            continue
        similarity = float(vectors[i] @ vectors[j])
        if similarity >= threshold:
            pairs.append((claims[i], claims[j], similarity))

    pairs.sort(key=lambda p: -p[2])
    log.info("claims.pairs", considered=len(claims), candidates=len(pairs), kept=min(len(pairs), limit))
    return pairs[:limit]


def relate_claims(
    client: LLMClient, pairs: list[tuple[Claim, Claim, float]]
) -> list[Relation]:
    """Label every candidate pair in one batched call."""
    if not pairs:
        return []

    listing = "\n\n".join(
        f"[{index}]\n"
        f"  A ({a.document_title}): {a.text}\n"
        f"  B ({b.document_title}): {b.text}"
        for index, (a, b, _) in enumerate(pairs)
    )

    try:
        completion = client.complete(
            prompt=f"Claim pairs:\n\n{listing}\n\nLabel all {len(pairs)} pairs.",
            stage=Stage.contrast,
            tier=Tier.utility,
            system=RELATE_SYSTEM,
            schema=RELATE_SCHEMA,
            max_tokens=4000,
            cache_system=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("claims.relate_failed", error=str(exc))
        return []

    relations = []
    for entry in (completion.parsed or {}).get("relations", []):
        index = entry.get("pair_index")
        if not isinstance(index, int) or not 0 <= index < len(pairs):
            continue
        label = entry.get("relation", "unrelated")
        if label not in RELATIONS or label == "unrelated":
            continue
        a, b, similarity = pairs[index]
        relations.append(
            Relation(
                source=a,
                target=b,
                relation=label,
                rationale=str(entry.get("rationale", ""))[:400],
                similarity=similarity,
            )
        )
    log.info("claims.related", labelled=len(relations), of=len(pairs))
    return relations


def build_graph(
    client: LLMClient, documents: list[tuple[uuid.UUID, str, list]]
) -> ClaimGraph:
    """Extract claims from every document, then relate them across documents."""
    claims: list[Claim] = []
    for document_id, title, chunks in documents:
        claims.extend(
            extract_claims(client, document_id=document_id, document_title=title, chunks=chunks)
        )

    graph = ClaimGraph(claims=claims)
    graph.relations = relate_claims(client, candidate_pairs(claims))
    log.info("claims.graph", **graph.summary())
    return graph
