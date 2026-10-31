"""Second, independent relevance labels for the ranking harness.

`relevance.jsonl` is built by a keyword rubric over topics and dependency names
(`build_eval_sets.py`), so it rewards rankers that think in the same keywords. These
labels come from a model reading each persona's plain-language description and the
items, with no rubric, no topic tags and no sight of any ranker's output. Agreement on
both sets is better evidence than a gain on one.

One call per persona, recorded with model and prompt so the set can be regenerated.

Usage:
    python scripts/build_judge_labels.py
"""

from __future__ import annotations

import json
import pathlib
import re

import anthropic

from voicebrief.config import get_settings
from voicebrief.evalkit.annotation import PERSONAS

DATASETS = pathlib.Path(__file__).resolve().parents[1] / "eval" / "datasets"
MODEL = "claude-sonnet-5-5"

PROMPT = """\
You are labelling a test set for a personalised daily tech brief.

The listener: {description}

Below are {n} items from today's feeds. For each, decide: would this listener want it
as one of about six stories in their daily brief, because it bears on their work or
the interests described? Be selective, as a thoughtful colleague curating for them
would be. A generally interesting item outside their work is not relevant.

{items}

Reply with JSON only: {{"relevant": [the numbers of the relevant items]}}"""


def main() -> None:
    corpus = {
        r["id"]: r
        for r in map(json.loads, (DATASETS / "corpus.jsonl").read_text("utf-8").splitlines())
        if r
    }
    rubric = [
        json.loads(line)
        for line in (DATASETS / "relevance.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    key = get_settings().anthropic_api_key
    assert key, "ANTHROPIC_API_KEY is required"
    client = anthropic.Anthropic(api_key=key.get_secret_value())

    rows = []
    for persona in PERSONAS:
        ids = [
            r["item_id"] for r in rubric if r["persona"] == persona.key and r["item_id"] in corpus
        ]
        listing = "\n".join(
            f"{n}. {corpus[i]['title']} — {(corpus[i].get('summary') or '')[:280]}"
            for n, i in enumerate(ids, 1)
        )
        prompt = PROMPT.format(description=persona.description, n=len(ids), items=listing)
        reply = client.messages.create(
            model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in reply.content if b.type == "text")
        chosen = set(json.loads(re.search(r"\{.*\}", text, re.S).group(0))["relevant"])
        for n, item_id in enumerate(ids, 1):
            rows.append(
                {
                    "persona": persona.key,
                    "item_id": item_id,
                    "title": corpus[item_id]["title"],
                    "relevant": n in chosen,
                    "judge": MODEL,
                }
            )
        print(f"{persona.key:16} {len(chosen):>3} relevant / {len(ids)}")

    (DATASETS / "relevance_judge.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
