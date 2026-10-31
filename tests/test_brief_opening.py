"""How an episode opens: the daily brief goes straight to its agenda; a documents episode
keeps the cold open that frames the listener's own material."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voicebrief.graphs.brief import BriefGraph, Segment


@pytest.fixture
def written(monkeypatch):
    prompts = []

    def simple(self, prompt, kind, heading, max_tokens):
        prompts.append((kind, prompt))
        return Segment(position=0, kind=kind, heading=heading, script=f"{kind} text")

    def story(self, story, summary, position, budget, profile, language, style):
        return Segment(position=position, kind="story", heading=summary.headline, script="s")

    monkeypatch.setattr(BriefGraph, "_simple_segment", simple)
    monkeypatch.setattr(BriefGraph, "_write_story", story)

    def run(mode):
        prompts.clear()
        graph = BriefGraph(SimpleNamespace(), mode=mode, subject="my notes")
        summaries = [(None, SimpleNamespace(headline=f"Story {i}")) for i in range(3)]
        segments = graph.write_node({"summaries": summaries})["segments"]
        return segments, dict(prompts)

    return run


def test_the_daily_brief_starts_on_the_agenda(written):
    segments, prompts = written("brief")
    assert [s.kind for s in segments] == ["agenda", "story", "story", "story", "sign_off"]
    assert "cold_open" not in prompts
    assert "no greeting" in prompts["agenda"]


def test_story_positions_follow_the_agenda(written):
    segments, _ = written("brief")
    assert [s.position for s in segments if s.kind == "story"] == [1, 2, 3]


def test_a_documents_episode_keeps_its_cold_open(written):
    segments, _ = written("knowledge")
    assert [s.kind for s in segments][:2] == ["cold_open", "agenda"]
