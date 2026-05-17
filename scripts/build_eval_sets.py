"""Regenerate the clustering and relevance evaluation sets from the frozen corpus.

The dedup set is not regenerated here — its labels are pairwise human judgements and
live in `eval/datasets/dedup_pairs.jsonl` with a recorded reason per row.

Usage:
    python scripts/build_eval_sets.py
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

from voicebrief.evalkit.annotation import PERSONAS, is_relevant, topic_for

DATASETS = pathlib.Path(__file__).resolve().parents[1] / "eval" / "datasets"


def main() -> None:
    corpus = [
        json.loads(line)
        for line in (DATASETS / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # ── clustering labels ────────────────────────────────────────────────────
    cluster_rows = []
    for row in corpus:
        topic = topic_for(row["title"], row.get("summary", ""), row.get("source", ""))
        cluster_rows.append({"id": row["id"], "title": row["title"], "topic": topic})

    (DATASETS / "cluster_labels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cluster_rows) + "\n",
        encoding="utf-8",
    )

    distribution = Counter(r["topic"] for r in cluster_rows)
    labelled = sum(v for k, v in distribution.items() if k != "other")

    # ── relevance labels ─────────────────────────────────────────────────────
    # Sampled evenly across the corpus so the set is not dominated by whichever
    # source happened to publish most that day.
    by_topic: dict[str, list[dict]] = {}
    for row, cluster in zip(corpus, cluster_rows, strict=True):
        by_topic.setdefault(cluster["topic"], []).append(row)

    sample: list[dict] = []
    # Sized so the set clears the PRD target of 200 (persona, item) pairs.
    per_topic = max(1, 90 // max(len(by_topic), 1))
    for topic, rows in sorted(by_topic.items()):
        sample.extend(rows[:per_topic])

    relevance_rows = []
    for persona in PERSONAS:
        for row in sample:
            topic = topic_for(row["title"], row.get("summary", ""), row.get("source", ""))
            relevance_rows.append(
                {
                    "persona": persona.key,
                    "item_id": row["id"],
                    "title": row["title"],
                    "topic": topic,
                    "relevant": is_relevant(
                        persona, row["title"], row.get("summary", ""), topic
                    ),
                }
            )

    (DATASETS / "relevance.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in relevance_rows) + "\n",
        encoding="utf-8",
    )

    positives = Counter(r["persona"] for r in relevance_rows if r["relevant"])

    print(f"cluster_labels.jsonl : {len(cluster_rows)} items, {labelled} in a real topic")
    print(f"  topics: {len(distribution) - 1} + other({distribution['other']})")
    for topic, count in distribution.most_common():
        print(f"    {topic:22} {count}")
    print(f"\nrelevance.jsonl      : {len(relevance_rows)} (persona, item) pairs")
    for persona in PERSONAS:
        total = sum(1 for r in relevance_rows if r["persona"] == persona.key)
        print(f"    {persona.key:16} {positives[persona.key]:>3} relevant / {total}")


if __name__ == "__main__":
    main()
