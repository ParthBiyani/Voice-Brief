"""Mode 2: your own documents and repositories, turned into an episode.

The same machinery as the daily brief (summarise with citations, write one spoken segment
per story, verify every sentence against its source), pointed at material you supply
instead of the news. What is specific to this mode:

* A **story** is a group of consecutive chunks from one document, so a long paper becomes
  a few segments in reading order. For a repository the chunks are already ordered
  architecture-first (README, manifests, then modules by how much the rest of the code
  imports them), so the episode explains what the project *is* before the details.
* With two or more documents, claims are extracted from each and compared, and the places
  where they **contradict** become a dedicated "where the sources disagree" segment. That
  is the point of the mode: five summaries are less useful than the one fact on which two
  of the five disagree.
* Everything said is grounded in the supplied text and cited by document and section;
  nothing is read from the web.
"""

from __future__ import annotations

import io
import math
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from voicebrief.db.models import Claim as ClaimRow
from voicebrief.db.models import ClaimRelation as ClaimRelationRow
from voicebrief.db.models import Document
from voicebrief.documents.claims import ClaimGraph, build_graph
from voicebrief.documents.parse import (
    DOC_EXTENSIONS,
    SKIP_DIRS,
    SOURCE_EXTENSIONS,
    ParsedDocument,
    parse_repo_zip,
    parse_upload,
)
from voicebrief.graphs.brief import BriefGraph, BriefState, Segment
from voicebrief.llm.base import BudgetExceeded, Stage, Tier
from voicebrief.llm.client import LLMClient
from voicebrief.logging import get_logger
from voicebrief.pipeline.ranking import RankCandidate, RankedStory
from voicebrief.pipeline.summarize import SourceRef

log = get_logger(__name__)

CHUNKS_PER_STORY = 5
MAX_BODY_CHARS = 1500
MAX_DOCUMENTS = 8
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_CONTRASTS = 4
MAX_REPO_FILE_BYTES = 400_000


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def parse_directory(path: Path) -> ParsedDocument:
    """A repository (or any folder of code and notes) on disk, as one document.

    The folder is packed into an in-memory archive and handed to the same parser an
    uploaded zip goes through, so a local checkout and a downloaded one are treated
    identically.
    """
    root = path.resolve()
    buffer = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.stat().st_size > MAX_REPO_FILE_BYTES:
                continue
            relative = file.relative_to(root)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            if file.suffix.lower() not in SOURCE_EXTENSIONS | DOC_EXTENSIONS:
                continue
            archive.write(file, f"{root.name}/{relative.as_posix()}")
            count += 1
    if count == 0:
        raise ValueError(f"{path} contains no readable source or documentation files")
    return parse_repo_zip(buffer.getvalue(), root.name)


def ensure_substance(document: ParsedDocument) -> ParsedDocument:
    """Reject a document with nothing to explain, in words a person can act on.

    The parser skips fragments below its minimum size, which is right for a stray
    heading but means a very short note parses to nothing at all; an episode built from
    that would be silently empty.
    """
    if not document.chunks:
        raise ValueError(f"{document.title} has too little text to explain")
    return document


def load_documents(paths: list[Path]) -> list[ParsedDocument]:
    """Parse files and folders. Raises ValueError with a plain message on anything unusable."""
    if not paths:
        raise ValueError("give at least one document or folder")
    if len(paths) > MAX_DOCUMENTS:
        raise ValueError(f"at most {MAX_DOCUMENTS} documents per episode")

    documents: list[ParsedDocument] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        if path.is_dir():
            documents.append(ensure_substance(parse_directory(path)))
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path.name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        documents.append(ensure_substance(parse_upload(path.name, path.read_bytes())))
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Stories from documents
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DocumentStory:
    story: RankedStory
    document_id: uuid.UUID
    document_title: str


def stories_per_document(chunk_counts: list[int], max_stories: int) -> list[int]:
    """Spread the story budget across documents, every document getting at least one."""
    if not chunk_counts:
        return []
    wanted = [max(1, math.ceil(count / CHUNKS_PER_STORY)) for count in chunk_counts]
    budget = max(max_stories, len(chunk_counts))
    while sum(wanted) > budget:
        # Trim the document with the most stories first, never below one.
        biggest = max(range(len(wanted)), key=lambda i: wanted[i])
        if wanted[biggest] <= 1:
            break
        wanted[biggest] -= 1
    return wanted


def build_stories(
    documents: list[tuple[uuid.UUID, ParsedDocument]], max_stories: int
) -> list[DocumentStory]:
    counts = stories_per_document([len(d.chunks) for _, d in documents], max_stories)
    stories: list[DocumentStory] = []

    for (document_id, parsed), n_stories in zip(documents, counts, strict=True):
        chunks = parsed.chunks
        if not chunks:
            continue
        # Contiguous groups keep reading order; when there are fewer stories than the
        # natural grouping, each simply takes a larger share.
        size = max(1, math.ceil(len(chunks) / n_stories))
        for start in range(0, len(chunks), size):
            group = chunks[start : start + size][: CHUNKS_PER_STORY * 2]
            sources = []
            bodies = {}
            for chunk in group:
                item_id = uuid.uuid4()
                label = chunk.section or parsed.title
                sources.append(
                    SourceRef(
                        item_id=item_id,
                        title=f"{parsed.title}: {label}"[:200],
                        url=f"doc://{document_id}/{chunk.index}",
                        source_slug=parsed.kind,
                    )
                )
                bodies[item_id] = chunk.text[:MAX_BODY_CHARS]

            first = group[0].section or parsed.title
            title = parsed.title if len(chunks) <= size else f"{parsed.title}: {first}"
            candidate = RankCandidate(
                cluster_id=uuid.uuid4(),
                title=title,
                summary=group[0].text[:300],
                url=sources[0].url,
                sources=sources,
                bodies=bodies,
                source_slug=parsed.kind,
                cluster_size=len(group),
            )
            order = 1.0 - len(stories) / 100
            stories.append(
                DocumentStory(
                    story=RankedStory(candidate=candidate, score=order, heuristic_score=order),
                    document_id=document_id,
                    document_title=parsed.title,
                )
            )
    return stories[: max(max_stories, len(documents))]


# ─────────────────────────────────────────────────────────────────────────────
# Where the sources disagree
# ─────────────────────────────────────────────────────────────────────────────
CONTRAST_SYSTEM = """\
You write one short spoken segment of an audio episode: the places where the listener's
own documents disagree with each other.

You are given pairs of claims, each from a named document, that were judged to
contradict. For each pair say, plainly, who says what.

Rules:
- Use only the claims given. Do not decide who is right and do not add context.
- Name the documents by title. Quote the substance of each claim faithfully.
- Short spoken sentences. No lists, headings or markdown.
- Never read a URL aloud.\
"""


def contrast_segment(
    client: LLMClient, graph: ClaimGraph, *, language: str = "en"
) -> Segment | None:
    """A segment on the strongest contradictions, or None when the sources agree."""
    contradictions = [
        r for r in graph.contradictions if r.source.document_id != r.target.document_id
    ][:MAX_CONTRASTS]
    if not contradictions:
        return None

    blocks = []
    citations: list[dict[str, str]] = []
    for number, relation in enumerate(contradictions, start=1):
        blocks.append(
            f"[{number}] {relation.source.document_title} says: {relation.source.text}\n"
            f"    {relation.target.document_title} says: {relation.target.text}\n"
            f"    why they conflict: {relation.rationale}"
        )
        for claim in (relation.source, relation.target):
            if any(c["item_id"] == str(claim.id) for c in citations):
                continue  # the same claim can appear in several conflicting pairs
            citations.append(
                {
                    "url": f"doc://{claim.document_id}/claim",
                    "title": f"{claim.document_title}: {claim.section or 'claim'}"[:200],
                    "item_id": str(claim.id),
                }
            )

    try:
        completion = client.complete(
            prompt="Contradicting claims:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nWrite the segment{' in Hindi' if language == 'hi' else ''}.",
            stage=Stage.contrast,
            tier=Tier.flagship,
            system=CONTRAST_SYSTEM,
            max_tokens=700,
            cache_system=True,
            fallback_tier=Tier.utility,
        )
    except BudgetExceeded:
        raise
    except Exception as exc:
        log.warning("knowledge.contrast_failed", error=str(exc)[:200])
        return None

    script = completion.text.strip()
    if not script:
        return None
    return Segment(
        position=0,
        kind="story",
        heading="Where your sources disagree",
        script=script,
        citations=citations,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def persist_documents(
    session, user_id: uuid.UUID, parsed: list[ParsedDocument], origins: list[str]
) -> list[tuple[uuid.UUID, ParsedDocument]]:
    """Store what was supplied (metadata only; the text lives in the segments' sources)."""
    pairs = []
    for document, origin in zip(parsed, origins, strict=True):
        row = Document(
            user_id=user_id,
            kind=document.kind,
            title=document.title,
            origin=origin,
            page_count=document.page_count,
            chunk_count=len(document.chunks),
            meta=dict(document.meta or {}),
        )
        session.add(row)
        session.flush()
        pairs.append((row.id, document))
    return pairs


def persist_claims(session, graph: ClaimGraph) -> None:
    for claim in graph.claims:
        session.add(
            ClaimRow(
                id=claim.id,
                document_id=claim.document_id,
                text=claim.text,
                section=claim.section,
                confidence=claim.confidence,
            )
        )
    session.flush()
    for relation in graph.relations:
        if relation.relation == "unrelated":
            continue
        session.add(
            ClaimRelationRow(
                source_claim_id=relation.source.id,
                target_claim_id=relation.target.id,
                relation=relation.relation,
                rationale=relation.rationale,
            )
        )
    session.flush()


# ─────────────────────────────────────────────────────────────────────────────
# The flow
# ─────────────────────────────────────────────────────────────────────────────
def run_knowledge(
    client: LLMClient,
    session,
    documents: list[tuple[uuid.UUID, ParsedDocument]],
    *,
    language: str = "en",
    style: str = "solo_anchor",
    max_stories: int = 6,
    target_minutes: int = 10,
) -> dict:
    """Summarise, contrast, write and verify. Returns the same shape as the brief graph."""
    subject = ", ".join(d.title for _, d in documents[:2]) + (
        f" and {len(documents) - 2} more" if len(documents) > 2 else ""
    )
    graph = BriefGraph(
        client,
        max_stories=max_stories,
        target_minutes=target_minutes,
        mode="knowledge",
        subject=subject,
    )

    stories = build_stories(documents, max_stories)
    state: BriefState = {
        "selected": [s.story for s in stories],
        "profile": None,
        "declared_topics": set(),
        "language": language,
        "style": style,
        "errors": [],
    }

    claim_graph: ClaimGraph | None = None
    if len(documents) >= 2:
        try:
            claim_graph = build_graph(
                client, [(doc_id, parsed.title, parsed.chunks) for doc_id, parsed in documents]
            )
            persist_claims(session, claim_graph)
        except BudgetExceeded:
            raise
        except Exception as exc:
            log.warning("knowledge.claims_failed", error=f"{type(exc).__name__}: {exc}"[:200])
            state["errors"].append("could not compare the documents' claims")

    state.update(cast(BriefState, graph.summarize_node(state)))
    state.update(cast(BriefState, graph.write_node(state)))

    contrast = contrast_segment(client, claim_graph, language=language) if claim_graph else None
    if contrast is not None:
        segments = state.get("segments", [])
        # Before the sign-off, after the document stories.
        state["segments"] = [*segments[:-1], contrast, *segments[-1:]] if segments else [contrast]

    state.update(cast(BriefState, graph.assemble_node(state)))
    state.update(cast(BriefState, graph.verify_node(state)))

    episode = state.get("episode", {})
    # The synthetic ids tying a segment to its summary were only needed for verification.
    # A segment's cluster id is a foreign key to a news cluster, which documents do not have.
    for segment in episode.get("segments", []):
        segment.cluster_id = None
    episode["errors"] = state.get("errors", [])
    episode["cost_inr"] = round(client.run_cost_inr, 4)
    episode["ranking"] = []
    # How many story segments were planned: a short document yields one, however high the
    # ceiling, and that is not an incomplete episode.
    episode["planned_stories"] = len(stories) + (1 if contrast is not None else 0)
    if claim_graph is not None:
        episode["claims"] = claim_graph.summary()
    return episode


__all__ = [
    "DocumentStory",
    "build_stories",
    "contrast_segment",
    "ensure_substance",
    "load_documents",
    "parse_directory",
    "persist_claims",
    "persist_documents",
    "run_knowledge",
    "stories_per_document",
]
