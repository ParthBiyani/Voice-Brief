"""Write the same brief with different models and compare them side by side.

Nothing is stored, voiced or delivered: the stories are the ones today's `daily` would
see, each configuration writes its own script from them, and the database transaction is
rolled back at the end, so the ledger and history are untouched. The budget ceiling is
lifted for the run on purpose: this is a measurement of quality against cost, and the
budget can then be set from the result.

    python scripts/compare_models.py                 # three configurations
    python scripts/compare_models.py --only sonnet   # one of them
    python scripts/compare_models.py --stories 6 --minutes 9

The report is written to eval/runs/ as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from voicebrief.db import session_scope
from voicebrief.graphs.brief import BriefGraph
from voicebrief.llm.anthropic_provider import AnthropicProvider
from voicebrief.llm.client import LLMClient
from voicebrief.pipeline.daily import (
    DEFAULT_MINUTES,
    DEFAULT_STORIES,
    BriefRequest,
    resolve_personal,
    resolve_profile,
    resolve_user,
    seed_terms,
)

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"

# name -> (model that searches/ranks/summarises, model that writes the script)
CONFIGS = {
    "current": (HAIKU, SONNET),
    "sonnet": (SONNET, SONNET),
    "haiku": (HAIKU, HAIKU),
}

# Words that mark machine-sounding prose; fewer is better.
AI_TELLS = re.compile(
    r"\b(delve|landscape|realm|game[- ]changer|revolutioni[sz]e|unlock|leverage|seamless"
    r"|robust|cutting[- ]edge|notably|importantly|it'?s worth noting|let'?s unpack|dive into)\b",
    re.IGNORECASE,
)
DIGITS = re.compile(r"\d")


def run_one(name: str, rank_model: str, script_model: str, args, session, context) -> dict:
    provider = AnthropicProvider(flagship_model=script_model, utility_model=rank_model)
    client = LLMClient(session, provider, durable_ledger=False, daily_budget_inr=1e9)
    graph = BriefGraph(client, max_stories=args.stories, target_minutes=args.minutes)
    episode = graph.run(
        context["candidates"],
        profile=context["profile"],
        declared_topics=context["topics"],
        language=context["language"],
        style=context["style"],
        affinity=context["affinity"],
    )
    segments = episode.get("segments", [])
    scripts = "\n".join(s.script for s in segments)
    stories = [s for s in segments if s.kind == "story"]
    ranking = [r for r in episode.get("ranking", []) if r["picked"]]
    return {
        "name": name,
        "rank_model": rank_model,
        "script_model": script_model,
        "cost_inr": episode["cost_inr"],
        "words": episode["word_count"],
        "minutes": episode["estimated_minutes"],
        "attribution": episode["grounding"]["attribution_rate"],
        "stories": len(stories),
        "picked": [
            {"title": r["title"][:90], "source": r["source_slug"], "llm_score": r["llm_score"]}
            for r in ranking
        ],
        "ai_tells": len(AI_TELLS.findall(scripts)),
        "digits_in_script": len(DIGITS.findall(scripts)),
        "errors": episode.get("errors", []),
        "script": [{"kind": s.kind, "heading": s.heading, "text": s.script} for s in segments],
    }


def markdown(results: list[dict], when: str) -> str:
    lines = [f"# Model comparison, {when}", ""]
    lines += [
        "| config | rank + summarise | script | cost | words | sourced | AI-tell words |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['rank_model']} | {r['script_model']} | ₹{r['cost_inr']:.2f} "
            f"| {r['words']} | {r['attribution']:.2f} | {r['ai_tells']} |"
        )
    for r in results:
        lines += ["", f"## {r['name']}", "", "Picked:"]
        lines += [f"- {p['title']} ({p['source']}, model score {p['llm_score']})" for p in r["picked"]]
        for part in r["script"][:3]:
            lines += ["", f"**{part['kind']}: {part['heading']}**", "", part["text"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--stories", type=int, default=DEFAULT_STORIES)
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    args = parser.parse_args()

    from voicebrief.pipeline.candidates import CandidateReport, build_candidates

    with session_scope() as session:
        request = BriefRequest()
        user = resolve_user(session, request)
        profile = resolve_profile(session, user)
        try:
            from voicebrief.pipeline.daily import attach_personal

            personal = resolve_personal(session, user, seeds=seed_terms(profile))
            profile = attach_personal(profile, personal, login=user.github_login)
        except Exception as exc:
            print(f"personal context skipped: {exc}", file=sys.stderr)
        from voicebrief.personalization.feedback import load_affinity

        context = {
            "candidates": build_candidates(session, user_id=user.id, report=CandidateReport(0)),
            "profile": profile,
            "topics": set(user.topics or []),
            "language": user.language.value,
            "style": user.style.value,
            "affinity": load_affinity(session, user.id),
        }
        if not context["candidates"]:
            print("no fresh stories: run `voicebrief ingest` first", file=sys.stderr)
            return 1

        results = []
        for name in args.only or list(CONFIGS):
            rank_model, script_model = CONFIGS[name]
            print(f"running {name} ({rank_model} ranks, {script_model} writes)...", flush=True)
            results.append(run_one(name, rank_model, script_model, args, session, context))
            r = results[-1]
            print(f"  cost ₹{r['cost_inr']:.2f}, {r['words']} words, sourced {r['attribution']:.2f}")
        session.rollback()  # nothing from this run is kept

    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = Path(__file__).resolve().parent.parent / "eval" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"models-{datetime.now():%Y%m%d-%H%M}"
    (out / f"{stem}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / f"{stem}.md").write_text(markdown(results, when), encoding="utf-8")
    print(f"report: eval/runs/{stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
