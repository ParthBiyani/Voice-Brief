"""Mode 2: documents and repositories to an episode."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from voicebrief.api.main import app
from voicebrief.db.models import Claim as ClaimRow
from voicebrief.db.models import (
    ClaimRelation,
    Document,
    Episode,
    EpisodeMode,
    EpisodeStatus,
    User,
)
from voicebrief.db.session import get_engine
from voicebrief.documents.claims import Claim, ClaimGraph, Relation
from voicebrief.documents.parse import Chunk, ParsedDocument
from voicebrief.graphs import knowledge
from voicebrief.graphs.knowledge import (
    build_stories,
    contrast_segment,
    load_documents,
    parse_directory,
    persist_documents,
    run_knowledge,
    stories_per_document,
)
from voicebrief.llm.base import BudgetExceeded, Stage
from voicebrief.pipeline.episodes import judge_episode

NOTE = "# Notes\n\n" + "Qdrant is fast and supports payload filtering natively. " * 4


def document(title="Paper", n_chunks=4, kind="pdf") -> ParsedDocument:
    return ParsedDocument(
        title=title,
        kind=kind,
        chunks=[
            Chunk(
                index=i,
                text=f"{title} section {i} says something specific.",
                section=f"S{i}",
                kind="section",
            )
            for i in range(n_chunks)
        ],
    )


class TestBudget:
    def test_every_document_gets_at_least_one_story(self):
        assert min(stories_per_document([1, 1, 1], 1)) == 1

    def test_a_short_document_is_one_story_a_long_one_several(self):
        assert stories_per_document([3, 25], 10) == [1, 5]

    def test_the_budget_trims_the_longest_document_first(self):
        counts = stories_per_document([25, 25, 5], 4)
        assert sum(counts) <= 4 or counts == [1, 1, 1]
        assert counts[2] == 1

    def test_no_documents_no_stories(self):
        assert stories_per_document([], 6) == []


class TestStories:
    def pairs(self, *docs):
        return [(uuid.uuid4(), d) for d in docs]

    def test_one_story_per_short_document_in_reading_order(self):
        stories = build_stories(self.pairs(document("A", 3), document("B", 3)), 6)
        assert [s.document_title for s in stories] == ["A", "B"]

    def test_a_long_document_becomes_consecutive_groups(self):
        stories = build_stories(self.pairs(document("Long", 20)), 4)
        sections = [[src.title for src in s.story.candidate.sources] for s in stories]
        flat = [name for group in sections for name in group]
        assert flat == sorted(flat, key=lambda t: int(t.split(": S")[1]))  # reading order kept
        assert len(stories) <= 4

    def test_sources_are_cited_by_document_and_section_not_by_url(self):
        story = build_stories(self.pairs(document("Paper", 2)), 3)[0].story.candidate
        assert all(src.url.startswith("doc://") for src in story.sources)
        assert story.sources[0].title.startswith("Paper: S0")
        assert set(story.bodies) == {src.item_id for src in story.sources}

    def test_each_story_gets_its_own_identity(self):
        stories = build_stories(self.pairs(document("A", 12)), 6)
        ids = [s.story.candidate.cluster_id for s in stories]
        assert len(set(ids)) == len(ids)

    def test_long_chunks_are_bounded(self):
        big = ParsedDocument("Big", "text", [Chunk(0, "x" * 10_000, "body", "section")])
        body = next(
            iter(build_stories([(uuid.uuid4(), big)], 3)[0].story.candidate.bodies.values())
        )
        assert len(body) <= knowledge.MAX_BODY_CHARS

    def test_a_document_with_no_chunks_is_skipped_without_error(self):
        empty = ParsedDocument("Empty", "text", [])
        stories = build_stories([(uuid.uuid4(), empty), (uuid.uuid4(), document("A", 2))], 4)
        assert stories[0].document_title == "A"


class TestLoading:
    def test_a_folder_is_read_as_a_repository_with_the_readme_first(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "# Widget\n\nWidget is a small tool that does one thing well. "
            + "It is documented here. " * 8
        )
        (tmp_path / "app.py").write_text(
            'from lib import helper\n\n\ndef main():\n    """Entry point."""\n    return helper() + 1\n'
            * 3
        )
        (tmp_path / "lib.py").write_text(
            'def helper():\n    """Shared by the entry point."""\n    return 41 + 1 + 0 + 0\n' * 3
        )
        parsed = parse_directory(tmp_path)
        assert parsed.kind == "repo"
        assert parsed.chunks[0].kind == "readme" and "Widget" in parsed.chunks[0].text
        files = [c.section for c in parsed.chunks if c.kind == "file"]
        # lib.py is imported by app.py, so it is the more central module and comes first.
        assert files.index("lib.py") < files.index("app.py")

    def test_dependency_and_build_folders_are_ignored(self, tmp_path):
        (tmp_path / "README.md").write_text("# P\n\n" + "A readable project description. " * 6)
        for ignored in ("node_modules", ".git", "__pycache__", ".venv"):
            (tmp_path / ignored).mkdir()
            (tmp_path / ignored / "junk.py").write_text("SECRET_INTERNAL_MARKER = 1\n" * 10)
        text = " ".join(c.text for c in parse_directory(tmp_path).chunks)
        assert "SECRET_INTERNAL_MARKER" not in text

    def test_oversized_and_unreadable_file_types_are_skipped(self, tmp_path):
        (tmp_path / "README.md").write_text("# P\n\n" + "A readable project description. " * 6)
        (tmp_path / "image.png").write_bytes(b"\x89PNG...." * 50)
        (tmp_path / "huge.py").write_text("x = 1\n" * 100_000)
        text = " ".join(c.text for c in parse_directory(tmp_path).chunks)
        assert "PNG" not in text and text.count("x = 1") < 100

    def test_an_empty_folder_is_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="no readable"):
            parse_directory(tmp_path)

    def test_a_note_too_short_to_explain_is_rejected_not_turned_into_an_empty_episode(
        self, tmp_path
    ):
        tiny = tmp_path / "tiny.md"
        tiny.write_text("# Hi\n\nok")
        with pytest.raises(ValueError, match="too little text"):
            load_documents([tiny])

    def test_a_markdown_file_loads(self, tmp_path):
        note = tmp_path / "notes.md"
        note.write_text(NOTE)
        assert load_documents([note])[0].title == "notes"

    def test_problems_are_reported_in_plain_words(self, tmp_path):
        with pytest.raises(ValueError, match="at least one"):
            load_documents([])
        with pytest.raises(ValueError, match="does not exist"):
            load_documents([tmp_path / "missing.md"])
        odd = tmp_path / "data.xyz"
        odd.write_text("x")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_documents([odd])
        with pytest.raises(ValueError, match="at most"):
            load_documents([Path(f"{i}.md") for i in range(knowledge.MAX_DOCUMENTS + 1)])


def claim(doc, title, text) -> Claim:
    return Claim(
        id=uuid.uuid4(), document_id=doc, document_title=title, text=text, section="Reasons"
    )


def conflict(a_doc, b_doc, rationale="they say opposite things"):
    a = claim(a_doc, "For", "Upgrades are predictable")
    b = claim(b_doc, "Against", "Upgrades are risky")
    return ClaimGraph([a, b], [Relation(a, b, "contradicts", rationale)]), a, b


class FakeClient:
    """Scripted by stage; the real client's ledger and budget are tested elsewhere."""

    run_cost_inr = 0.0

    def __init__(self, fail_on=None, fail_with=None):
        self.stages = []
        self.fail_on = fail_on
        self.fail_with = fail_with

    def complete(self, *, stage, **kwargs):
        self.stages.append(stage)
        if self.fail_on is stage:
            raise self.fail_with
        if stage is Stage.cluster_summary:
            return SimpleNamespace(
                parsed={
                    "headline": "The documents explain a thing",
                    "what_changed": "A thing is explained.",
                    "why_it_matters": "",
                    "key_facts": [
                        {"fact": "The material says something specific.", "source_index": 0}
                    ],
                },
                text="",
            )
        return SimpleNamespace(parsed=None, text=f"A spoken script for the {stage.value} stage.")


class TestContrast:
    def test_a_real_disagreement_becomes_a_segment_citing_both_claims(self):
        graph, a, b = conflict(uuid.uuid4(), uuid.uuid4())
        segment = contrast_segment(FakeClient(), graph)
        assert segment is not None and segment.heading == "Where your sources disagree"
        assert {c["item_id"] for c in segment.citations} == {str(a.id), str(b.id)}

    def test_the_same_claim_is_cited_once_even_in_several_conflicts(self):
        d1, d2 = uuid.uuid4(), uuid.uuid4()
        a = claim(d1, "For", "Upgrades are predictable")
        b = claim(d2, "Against", "Upgrades are risky")
        c = claim(d2, "Against", "Upgrades have already broken a setup")
        graph = ClaimGraph(
            [a, b, c], [Relation(a, b, "contradicts", "x"), Relation(a, c, "contradicts", "y")]
        )
        assert len(contrast_segment(FakeClient(), graph).citations) == 3

    def test_agreement_means_no_segment_and_no_model_call(self):
        d1, d2 = uuid.uuid4(), uuid.uuid4()
        a, b = claim(d1, "A", "x"), claim(d2, "B", "x too")
        client = FakeClient()
        assert (
            contrast_segment(client, ClaimGraph([a, b], [Relation(a, b, "agrees", "same")])) is None
        )
        assert client.stages == []

    def test_a_contradiction_inside_one_document_is_not_a_disagreement_between_sources(self):
        same = uuid.uuid4()
        graph, *_ = conflict(same, same)
        assert contrast_segment(FakeClient(), graph) is None

    def test_a_model_failure_costs_the_segment_not_the_episode(self):
        client = FakeClient(fail_on=Stage.contrast, fail_with=RuntimeError("model declined"))
        graph, *_ = conflict(uuid.uuid4(), uuid.uuid4())
        assert contrast_segment(client, graph) is None

    def test_running_out_of_budget_stops_everything(self):
        client = FakeClient(
            fail_on=Stage.contrast, fail_with=BudgetExceeded("daily budget exhausted")
        )
        graph, *_ = conflict(uuid.uuid4(), uuid.uuid4())
        with pytest.raises(BudgetExceeded):
            contrast_segment(client, graph)


@pytest.mark.integration
class TestFlow:
    @pytest.fixture
    def session(self):
        with Session(get_engine()) as s:
            s.begin()
            try:
                yield s
            finally:
                s.rollback()

    @pytest.fixture
    def user(self, session):
        u = User(email=f"docs-{uuid.uuid4().hex[:8]}@example.com")
        session.add(u)
        session.flush()
        return u

    def stored(self, session, user, *docs):
        return persist_documents(session, user.id, list(docs), [f"{d.title}.md" for d in docs])

    def test_documents_are_stored_with_their_shape(self, session, user):
        pairs = self.stored(session, user, document("A", 4), document("B", 7))
        rows = session.execute(select(Document).where(Document.user_id == user.id)).scalars().all()
        assert {r.title: r.chunk_count for r in rows} == {"A": 4, "B": 7}
        assert pairs[0][0] == next(r.id for r in rows if r.title == "A")

    def test_two_documents_get_contrast_claims_stored_and_a_title_naming_them(
        self, session, user, monkeypatch
    ):
        pairs = self.stored(session, user, document("For", 2), document("Against", 2))
        graph, a, b = conflict(pairs[0][0], pairs[1][0])
        monkeypatch.setattr(knowledge, "build_graph", lambda client, docs: graph)

        episode = run_knowledge(FakeClient(), session, pairs, max_stories=4)

        kinds = [s.kind for s in episode["segments"]]
        headings = [s.heading for s in episode["segments"]]
        assert kinds[0] == "cold_open" and kinds[-1] == "sign_off"
        assert headings.index("Where your sources disagree") == len(headings) - 2
        assert episode["title"].startswith("For, Against")
        assert episode["claims"]["claims"] == 2
        # Scoped to this test's documents: the database is shared with real Mode 2 use.
        doc_ids = [pairs[0][0], pairs[1][0]]
        stored_claims = (
            session.execute(select(ClaimRow).where(ClaimRow.document_id.in_(doc_ids)))
            .scalars()
            .all()
        )
        assert {c.text for c in stored_claims} >= {a.text, b.text}
        relations = (
            session.execute(
                select(ClaimRelation).where(
                    ClaimRelation.source_claim_id.in_([c.id for c in stored_claims])
                )
            )
            .scalars()
            .all()
        )
        assert [r.relation for r in relations] == ["contradicts"]

    def test_segments_carry_no_cluster_ids_because_documents_have_no_clusters(self, session, user):
        pairs = self.stored(session, user, document("Solo", 3))
        episode = run_knowledge(FakeClient(), session, pairs, max_stories=3)
        assert all(s.cluster_id is None for s in episode["segments"])

    def test_one_document_means_no_claim_comparison_at_all(self, session, user, monkeypatch):
        called = []
        monkeypatch.setattr(knowledge, "build_graph", lambda *a, **k: called.append(1))
        pairs = self.stored(session, user, document("Solo", 3))
        episode = run_knowledge(FakeClient(), session, pairs, max_stories=3)
        assert called == [] and "claims" not in episode
        assert "Where your sources disagree" not in [s.heading for s in episode["segments"]]

    def test_a_failed_comparison_still_produces_the_episode(self, session, user, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("embedding model unavailable")

        monkeypatch.setattr(knowledge, "build_graph", boom)
        pairs = self.stored(session, user, document("A", 2), document("B", 2))
        episode = run_knowledge(FakeClient(), session, pairs, max_stories=4)
        assert any(s.kind == "story" for s in episode["segments"])
        assert any("compare" in e for e in episode["errors"])

    def test_the_whole_episode_is_stored_in_knowledge_mode(self, monkeypatch):
        from voicebrief.pipeline import documents_episode as de

        seen = {}

        def fake_flow(client, sess, stored, **kwargs):
            seen.update(kwargs)
            return run_knowledge(FakeClient(), sess, stored, **kwargs)

        monkeypatch.setattr(de, "run_knowledge", fake_flow)
        engine = get_engine()
        email = f"flow-{uuid.uuid4().hex[:8]}@example.com"
        try:
            with Session(engine) as work:
                result = de.generate_documents_episode(
                    work,
                    de.DocumentsRequest(email=email, render_audio=False, stories=3),
                    [document("Notes", 3)],
                    ["notes.md"],
                )
                work.commit()
            with Session(engine) as check:
                episode = check.get(Episode, result.episode_id)
                assert episode.mode is EpisodeMode.knowledge
                # One short document is one story, and that is a complete episode.
                assert episode.status is EpisodeStatus.ready
                assert len(episode.segments) >= 3
                assert check.query(Document).filter(Document.title == "Notes").count() >= 1
        finally:
            with Session(engine) as cleanup:
                ids = [u.id for u in cleanup.query(User).filter(User.email == email)]
                cleanup.query(Episode).filter(Episode.user_id.in_(ids)).delete(
                    synchronize_session=False
                )
                cleanup.query(Document).filter(Document.user_id.in_(ids)).delete(
                    synchronize_session=False
                )
                cleanup.query(User).filter(User.email == email).delete()
                cleanup.commit()
        assert seen["max_stories"] == 3


def test_one_short_document_is_a_complete_episode_not_an_incomplete_one():
    # Three stories were allowed, one was warranted, one was written: that is complete.
    status, reasons = judge_episode(
        story_count=1,
        requested_stories=1,
        render_audio=False,
        audio_key=None,
        duration_seconds=0.0,
        engine_name="none",
        configured_engine="null",
    )
    assert status is EpisodeStatus.ready and not reasons


class TestUploadApi:
    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr("voicebrief.api.extras.record_interaction", lambda *a, **k: None)

    def post(self, files, **data):
        return TestClient(app).post("/documents/podcast", files=files, data=data)

    def note(self, text="A reasonably long note about vector stores and indexes. ") -> bytes:
        return ("# A note\n\n" + text * 5).encode()

    def test_documents_are_accepted_parsed_and_handed_to_the_worker(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "voicebrief.api.extras._run_documents",
            lambda episode_id, request, parsed, origins: seen.update(
                parsed=parsed, origins=origins
            ),
        )
        response = self.post(
            [
                (
                    "files",
                    ("a.md", self.note("First note about Qdrant filtering. "), "text/markdown"),
                ),
                (
                    "files",
                    ("b.txt", self.note("Second note about pgvector indexes. "), "text/plain"),
                ),
            ],
            language="hi",
        )
        assert response.status_code == 202
        body = response.json()
        assert [d["title"] for d in body["documents"]] == ["a", "b"]
        assert body["stream_url"].endswith("/stream")
        assert seen["origins"] == ["a.md", "b.txt"]

    def test_an_unsupported_file_type_is_rejected_with_the_reason(self):
        response = self.post([("files", ("data.xyz", b"x", "application/octet-stream"))])
        assert response.status_code == 422
        assert "Unsupported file type" in response.json()["detail"]

    def test_a_corrupt_pdf_is_rejected_not_a_server_error(self):
        response = self.post([("files", ("bad.pdf", b"this is not a pdf", "application/pdf"))])
        assert response.status_code == 422 and "bad.pdf" in response.json()["detail"]

    def test_a_note_too_short_to_explain_is_rejected(self):
        response = self.post([("files", ("tiny.md", b"# Hi\n\nok", "text/markdown"))])
        assert response.status_code == 422 and "too little text" in response.json()["detail"]

    def test_too_many_files_are_rejected(self):
        files = [("files", (f"{i}.md", self.note(), "text/markdown")) for i in range(9)]
        assert self.post(files).status_code == 422

    def test_oversized_uploads_are_rejected(self, monkeypatch):
        monkeypatch.setattr("voicebrief.graphs.knowledge.MAX_FILE_BYTES", 10)
        response = self.post([("files", ("big.md", b"x" * 100, "text/markdown"))])
        assert response.status_code == 413

    def test_an_unknown_language_is_rejected(self):
        response = self.post([("files", ("a.md", self.note(), "text/markdown"))], language="fr")
        assert response.status_code == 422

    def test_no_files_is_rejected(self):
        assert TestClient(app).post("/documents/podcast").status_code == 422
